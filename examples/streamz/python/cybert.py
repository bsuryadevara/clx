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

import gc
import signal
import sys
import time
import cudf
import dask
import torch
import pandas as pd
from streamz import Stream
from tornado import ioloop
from clx_streamz_tools import utils


def inference(gdf):
    # Messages will be received and run through cyBERT inferencing
    worker = dask.distributed.get_worker()
    batch_start_time = int(round(time.time()))
    result_size = gdf.shape[0]
    gdf = gdf[["message"]]
    print("Processing batch size: " + str(result_size))
    parsed_df, confidence_df = worker.data["cybert"].inference(gdf["message"])
    confidence_df = confidence_df.add_suffix("_confidence")
    parsed_df = pd.concat([parsed_df, confidence_df], axis=1)
    torch.cuda.empty_cache()
    gc.collect()
    return (parsed_df, batch_start_time, result_size)


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


def worker_init():
    # Initialization for each dask worker
    from clx.analytics.cybert import Cybert
    import confluent_kafka as ck

    worker = dask.distributed.get_worker()
    cy = Cybert()
    print(
        "Initializing Dask worker: "
        + str(worker)
        + " with cybert model. Model File: "
        + str(args.model)
        + " Label Map: "
        + str(args.label_map)
    )
    cy.load_local_model(args.model)
    worker.data["cybert"] = cy

    sink = config["sink"].lower()
    if sink == "kafka":
        import confluent_kafka as ck

        print("Producer conf: " + str(kafka_conf["producer_conf"]))
        producer = ck.Producer(kafka_conf["producer_conf"])
        worker.data["sink"] = producer
    elif sink == "elasticsearch":
        from elasticsearch import Elasticsearch

        #         from elasticsearch import AsyncElasticsearch
        #
        #         es = AsyncElasticsearch([config["elasticsearch_conf"]["host"]])
        es_conf = config["elasticsearch_conf"]

        #         from ssl import create_default_context
        #
        #
        #         context = create_default_context(cafile=es_conf["cafile"])
        #         es_client = Elasticsearch(
        #             es_conf["host"].split(','),
        #             http_auth=(es_conf['username'], es_conf['password']),
        #             scheme="https",
        #             port=es_conf["port"],
        #             ssl_context=context,
        #         )
        es_client = Elasticsearch(es_conf["hosts"].split(","), port=es_conf["port"])
    elif sink == "filesystem":
        print(
            "Streaming process will write the output to location '{}'".format(
                config["output_dir"]
            )
        )
    else:
        print(
            "No valid sink provided in the configuration file. Please provide kafka/elasticsearch/filesystem"
        )
        sys.exit(-1)
    print("Successfully initialized dask worker " + str(worker))


def start_stream():
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
    sink_dict = {"kafka": sink_to_kafka, "elasticsearch": sink_to_es}

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
        else:
            sink.transport.close()
        loop.stop()
