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

import time
import dask
from clx_streamz_tools import utils
from clx_streamz_tools import streamz_workflow


class DGADetectionWorkflow(streamz_workflow.StreamzWorkflow):
    def inference(self, gdf):
        # Messages will be received and run through DGA inferencing
        batch_start_time = int(round(time.time()))
        worker = dask.distributed.get_worker()
        result_size = gdf.shape[0]
        gdf = gdf[["message"]]
        # s_time = time.time()
        gdf["url"] = gdf.message.str.extract("query:\s([a-zA-Z\.\-\:\/\-0-9]+)")
        gdf["url"] = gdf.url.str.lower()
        gdf = gdf[gdf["url"].str.endswith(".arpa") == False]
        gdf = gdf.reset_index(drop=True)
        # e_time = time.time()
        # print("time taken by extract function {} sec".format(e_time - s_time))
        extracted_gdf = dns.parse_url(gdf["url"], req_cols={"domain", "suffix"})
        domain_series = extracted_gdf["domain"] + "." + extracted_gdf["suffix"]
        gdf["domain"] = domain_series.str.strip(".")
        # e_parse_time = time.time()
        # print("time taken by parse_url function {} sec".format(e_parse_time - e_time))
        dd = worker.data["dga_detector"]
        # s_time = time.time()
        preds = dd.predict(domain_series)
        # e_time = time.time()
        # print("time taken by predict function {} sec".format(e_time - s_time))
        gdf["dga_probability"] = preds
        gdf["insert_time"] = batch_start_time
        e_time = time.time()
        print("time taken by inference function {} sec".format(e_time - batch_start_time))
        return (gdf, batch_start_time, result_size)

    def worker_init(self):
        # Initialization for each dask worker
        from clx.analytics.dga_detector import DGADetector

        worker = dask.distributed.get_worker()
        dd = DGADetector()
        dga_model_filepath = os.path.join(self.args.model, 'pytorch_model.bin')
        print(
            "Initializing Dask worker: "
            + str(worker)
            + " with dga model. Model File: "
            + dga_model_filepath
        )
        dd.load_model(dga_model_filepath)
        # this dict can be used for adding more objects to distributed dask worker
        obj_dict = {"dga_detector": dd}
        worker = utils.init_dask_workers(worker, self.config, obj_dict)


if __name__ == "__main__":
    dga_detection = DGADetectionWorkflow()
    dga_detection.start()
