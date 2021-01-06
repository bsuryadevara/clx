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
import time
import dask
import cudf
import pandas as pd
from clx_streamz_tools import utils
from clx.dns import dns_extractor as dns
from clx_streamz_tools import streamz_workflow


class CybertWorkflow(streamz_workflow.StreamzWorkflow):
    def inference(self, input_gdf):
        # Messages will be received and run through cyBERT inferencing
        worker = dask.distributed.get_worker()
        batch_start_time = int(round(time.time()))
        result_size = input_gdf.shape[0]
        input_gdf = input_gdf[["message"]]
        print("Processing batch size: " + str(result_size))
        parsed_df, confidence_df = worker.data["cybert"].inference(input_gdf["message"])
        confidence_df = confidence_df.add_suffix("_confidence")
        
        parsed_df = pd.concat([parsed_df, confidence_df], axis=1)
        parsed_gdf = cudf.from_pandas(parsed_df)
        parsed_gdf['message'] = input_gdf['message']
        
        del input_gdf
        
        parsed_gdf["url.full"] = parsed_gdf['url.full'].str.lower()
        parsed_gdf = parsed_gdf[parsed_gdf["url.full"].str.endswith(".arpa") == False]
        parsed_gdf = parsed_gdf.reset_index(drop=True)
        
        tld_extract_gdf = dns.parse_url(parsed_gdf['url.full'], req_cols={"domain", "suffix"})
        domain_series = tld_extract_gdf["domain"] + "." + tld_extract_gdf["suffix"]
        parsed_gdf["domain"] = domain_series.str.strip(".")
        preds = worker.data["dga_detector"].predict(parsed_gdf["domain"])
        parsed_gdf["dga_probability"] = preds
        parsed_gdf["insert_time"] = batch_start_time
        return (parsed_gdf, batch_start_time, result_size)

    def worker_init(self):
        # Initialization for each dask worker
        from clx.analytics.cybert import Cybert
        from clx.analytics.dga_detector import DGADetector
        
        worker = dask.distributed.get_worker()
        cy = Cybert()
        cybert_model_filepath = os.path.join(self.args.model, 'cybert', 'pytorch_model.bin')
        cybert_labels_filepath = os.path.join(self.args.model, 'cybert', 'config.json')   
        print(
            "Initializing Dask worker: "
            + str(worker)
            + " with cybert model. Model File: "
            + cybert_model_filepath
            + " Label Map: "
            + str(cybert_labels_filepath)
        )
        cy.load_model(cybert_model_filepath, cybert_labels_filepath)
        
        dd = DGADetector()
        dga_model_filepath = os.path.join(self.args.model, 'dga', 'pytorch_model.bin')
        print(
            "Initializing Dask worker: "
            + str(worker)
            + " with dga model. Model File: "
            + dga_model_filepath
        )
        dd.load_model(self.args.model)
        
        # this dict can be used for adding more objects to distributed dask worker
        obj_dict = {"cybert": cy, "dga_detector": dd}
        worker = utils.init_dask_workers(worker, self.config, obj_dict)


if __name__ == "__main__":
    cybert = CybertWorkflow()
    cybert.start()
