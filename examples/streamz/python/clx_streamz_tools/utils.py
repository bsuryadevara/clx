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

import json
import time
import argparse
import yaml
import dask
import random
from distributed import Client
from elasticsearch import helpers
from dask_cuda import LocalCUDACluster


def create_dask_client():
    print("Creating local cuda cluster as no dask scheduler is provided.")
    cluster = LocalCUDACluster()
    client = Client(cluster)
    print(client)
    return client


def kafka_sink(output_topic, parsed_df):
    worker = dask.distributed.get_worker()
    producer = worker.data["sink"]
    json_str = parsed_df.to_json(orient="records", lines=True)
    json_recs = json_str.split("\n")
    for json_rec in json_recs:
        producer.produce(output_topic, json_rec)
    producer.flush()

def es_sink(config, parsed_df):
    worker = dask.distributed.get_worker()
    es = worker.data["sink"]
    parsed_df["_id"] = random.getrandbits(40) + parsed_df.index
    parsed_df['_id'] = parsed_df['_id'].astype('int64')
    parsed_df["_index"] = config["index"]
    json_str = parsed_df.to_json(orient="records")
    docs = json.loads(json_str)
    pb = helpers.parallel_bulk(es, docs, 
                               chunk_size=15000, 
                               thread_count=16, 
                               queue_size=16)

def calc_benchmark(processed_data, size_per_log):
    # Calculates benchmark for the streamz workflow
    t1 = int(round(time.time() * 1000))
    t2 = 0
    size = 0.0
    batch_count = 0
    # Find min and max time while keeping track of batch count and size
    for result in processed_data:
        (ts1, ts2, result_size) = (result[1], result[2], result[3])
        if ts1 == 0 or ts2 == 0:
            continue
        batch_count = batch_count + 1
        t1 = min(t1, ts1)
        t2 = max(t2, ts2)
        size += result_size * size_per_log
    time_diff = t2 - t1
    throughput_mbps = size / (1024.0 * time_diff) if time_diff > 0 else 0
    avg_batch_size = size / (1024.0 * batch_count) if batch_count > 0 else 0
    return (time_diff, throughput_mbps, avg_batch_size)

def load_yaml(yaml_file):
    """Returns a dictionary of a configuration contained in the given yaml file"""
    with open(yaml_file) as yaml_file:
        config_dict = yaml.safe_load(yaml_file)
    return config_dict

def parse_arguments():
    # Establish script arguments
    parser = argparse.ArgumentParser(
        description="Streamz and Dask. \
                     Data will be read from the input kafka topic, \
                     processed using clx streamz workflows."
    )
    parser.add_argument("-c", "--conf", help="Source and Sink configuration filepath")
    parser.add_argument("-g", "--group_id", default="streamz", help="Kafka group ID")
    parser.add_argument("-m", "--model", help="Model filepath")
    parser.add_argument("-l", "--label_map", help="Label map filepath")
    parser.add_argument(
        "--max_batch_size",
        default=1000,
        type=int,
        help="Max batch size to read from kafka",
    )
    parser.add_argument("--poll_interval", type=str, help="Polling interval (ex: 60s)")
    parser.add_argument(
        "--benchmark",
        help="Captures benchmark, including throughput estimates, with provided avg log size in KB. (ex: 500 or 0.1)",
        type=float,
    )
    args = parser.parse_args()
    return args