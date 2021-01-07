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
import cudf
import pandas as pd
from clx_streamz_tools import utils
from clx_streamz_tools import streamz_workflow


class CybertWorkflow(streamz_workflow.StreamzWorkflow):
    def inference(self, gdf):
        # Messages will be received and run through cyBERT inferencing
        worker = dask.distributed.get_worker()
        batch_start_time = int(round(time.time()))
        result_size = gdf.shape[0]
        gdf = gdf[["message"]]
        print("Processing batch size: " + str(result_size))
        parsed_df, confidence_df = worker.data["cybert"].inference(gdf["message"])
        confidence_df = confidence_df.add_suffix("_confidence")
        parsed_df = pd.concat([parsed_df, confidence_df], axis=1)
        parsed_df['message'] = gdf['message'].values_host 
        torch.cuda.empty_cache()
        gc.collect()
        return (parsed_df, batch_start_time, result_size)

    def worker_init(self):
        # Initialization for each dask worker
        from clx.analytics.cybert import Cybert

        worker = dask.distributed.get_worker()
        cy = Cybert()
        cybert_model_filepath = os.path.join(self.args.model, 'pytorch_model.bin')
        cybert_labels_filepath = os.path.join(self.args.model, 'config.json')   
        print(
            "Initializing Dask worker: "
            + str(worker)
            + " with cybert model. Model File: "
            + cybert_model_filepath
            + " Label Map: "
            + str(cybert_labels_filepath)
        )
        cy.load_model(cybert_model_filepath, cybert_labels_filepath)
        # this dict can be used for adding more objects to distributed dask worker
        obj_dict = {"cybert": cy}
        worker = utils.init_dask_workers(worker, self.config, obj_dict)


if __name__ == "__main__":
    cybert = CybertWorkflow()
    cybert.start()
