# Copyright (c) 2020, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import gc
import sys
import time
import dask
import torch
import cudf
import signal
from streamz import Stream
from tornado import ioloop
from clx_streamz_tools import utils
from clx.dns import dns_extractor as dns


def inference(gdf):
    # Messages will be received and run through DGA inferencing
    batch_start_time = int(round(time.time()))
    worker = dask.distributed.get_worker()
    result_size = gdf.shape[0]
    gdf = gdf[["message"]]
    # s_time = time.time()
    gdf["url"] = gdf.message.str.extract("query:\s([a-zA-Z\.\-\:\/\-0-9]+)")
    gdf["url"] = gdf.url.str.lower()
    gdf = gdf[gdf["url"].str.endswith(".arpa") == False]
    # e_time = time.time()
    # print("time taken by extract function {} sec".format(e_time - s_time))
    extracted_gdf = dns.parse_url(gdf["url"], req_cols={"domain", "suffix"})
    domain_series = extracted_gdf["domain"] + "." + extracted_gdf["suffix"]
    gdf["domain"] = domain_series.str.strip(".")
    dd = worker.data["dga_detector"]
    # s_time = time.time()
    preds = dd.predict(domain_series)
    # e_time = time.time()
    # print("time taken by predict function {} sec".format(e_time - s_time))
    gdf["dga_probability"] = preds
    gdf["insert_time"] = batch_start_time
    torch.cuda.empty_cache()
    gc.collect()
    e_time = time.time()
    print("time taken by inference function {} sec".format(e_time - batch_start_time))
    return (gdf, batch_start_time, result_size)


def sink_to_kafka(processed_data):
    # Prediction data will be published to provided Kafka producer
    utils.kafka_sink(kafka_conf["output_topic"], processed_data[0])
    return processed_data


def sink_to_es(processed_data):
    # Prediction data will be published to ElasticSearch cluster
    utils.es_sink(config["elasticsearch_conf"], processed_data[0])
    return processed_data


def sink_to_fs(processed_data):
    # Prediction data will be written to disk
    utils.fs_sink(config, processed_data[0])
    return processed_data


def worker_init():
    # Initialization for each dask worker
    from clx.analytics.dga_detector import DGADetector

    worker = dask.distributed.get_worker()
    dd = DGADetector()
    print(
        "Initializing Dask worker: "
        + str(worker)
        + " with dga model. Model File: "
        + str(args.model)
    )
    dd.load_model(args.model)
    worker.data["dga_detector"] = dd

    sink = config["sink"].lower()
    if sink == "kafka":
        import confluent_kafka as ck

        print("Producer conf: " + str(kafka_conf["producer_conf"]))
        producer = ck.Producer(kafka_conf["producer_conf"])
        worker.data["sink"] = producer
    elif sink == "elasticsearch":
        from elasticsearch import Elasticsearch

        es_conf = config["elasticsearch_conf"]
        #         from elasticsearch import AsyncElasticsearch
        #
        #         es = AsyncElasticsearch([config["elasticsearch_conf"]["host"]])
        es_client = Elasticsearch(
            [
                es_conf["url"].format(
                    es_conf["username"], es_conf["password"], es_conf["port"]
                )
            ],
            use_ssl=True,
            verify_certs=True,
            ca_certs=es_conf["ca_file"],
        )
        worker.data["sink"] = es_client
    elif sink == "filesystem":
        print(
            "Streaming process will write the output to location '{}'".format(
                config["output_dir"]
            )
        )
    else:
        print(
            "No valid sink provided in the configuration file. Please provide kafka/elasticsearch"
        )
        sys.exit(-1)

    print("Successfully initialized dask worker " + str(worker))


def signal_term_handler(signal, frame):
    # Receives signal and calculates benchmark if indicated in argument
    print("Exiting streamz script...")
    if args.benchmark:
        (time_diff, throughput_mbps, avg_batch_size) = utils.calc_benchmark(
            output, args.benchmark
        )
        print("*** BENCHMARK ***")
        print(
            "Job duration: {:.3f} secs, Throughput(mb/sec):{:.3f}, Avg. Batch size(mb):{:.3f}".format(
                time_diff, throughput_mbps, avg_batch_size
            )
        )
    sys.exit(0)


def start_stream():
    # Define the streaming pipeline.
    source = Stream.from_kafka_batched(
        kafka_conf["input_topic"],
        kafka_conf["consumer_conf"],
        poll_interval=args.poll_interval,
        # npartitions value varies based on kafka topic partitions configuration.
        npartitions=kafka_conf["n_partitions"],
        asynchronous=True,
        dask=True,
        engine="cudf",
        max_batch_size=args.max_batch_size,
    )
    sink = config["sink"]
    global output
    # If benchmark arg is True, use streamz to compute benchmark
    if args.benchmark:
        print("Benchmark will be calculated")
        output = (
            source.map(inference)
            .map(lambda x: (x[0], x[1], int(round(time.time())), x[2]))
            .map(sink_dict[sink])
            .gather()
            .sink_to_list()
        )
    else:
        output = source.map(inference).map(sink_dict[sink]).gather()

    source.start()


if __name__ == "__main__":
    # Parse arguments
    args = utils.parse_arguments()
    config = utils.load_yaml(args.conf)
    kafka_conf = config["kafka_conf"]
    sink_dict = {
        "kafka": sink_to_kafka,
        "elasticsearch": sink_to_es,
        "filesystem": sink_to_fs,
    }

    if not os.path.exists(config["output_dir"]):
        print("Creating output directory '{}'".format(config["output_dir"]))
        os.makedirs(config["output_dir"])

    # Handle script exit
    signal.signal(signal.SIGTERM, signal_term_handler)
    signal.signal(signal.SIGINT, signal_term_handler)

    client = utils.create_dask_client()
    client.run(worker_init)

    print("Consumer conf: " + str(kafka_conf["consumer_conf"]))

    loop = ioloop.IOLoop.current()
    loop.add_callback(start_stream)

    try:
        loop.start()
    except KeyboardInterrupt:
        worker = dask.distributed.get_worker()
        sink = worker.data["sink"]
        if config["sink"] == "kafka":
            sink.close()
        elif config["sink"] == "elasticsearch":
            sink.transport.close()
        else:
            pass
        loop.stop()
