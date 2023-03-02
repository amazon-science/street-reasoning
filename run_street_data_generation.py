# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


# 1 - Download AQUA-RAT OK
# 2 - Download AR-LSAT
# 3 - Get TLUs from AQUA-RAT
# 4 - Get TLUS from AR-LSAT
# 5 - Put together all the tasks

import re
import json
import urllib.request
from utils import dataset_utils
from pathlib import Path

############################################################
# Download Existing Datasets
############################################################

def download_original_datasets():
    print("Downloading original AQUA-RAT...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/deepmind/AQuA/master/train.json", 
        Path("data/aqua_rat/original_train.json")
    )
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/deepmind/AQuA/master/dev.json", 
        Path("data/aqua_rat/original_dev.json")
    )
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/deepmind/AQuA/master/test.json", 
        Path("data/aqua_rat/original_test.json")
    )

    print("Downloading original AR-LSAT...")
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/zhongwanjun/AR-LSAT/main/data/AR_TrainingData.json", 
        Path("data/ar_lsat/original_train.json")
    )
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/zhongwanjun/AR-LSAT/main/data/AR_DevelopmentData.json", 
        Path("data/ar_lsat/original_dev.json")
    )
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/zhongwanjun/AR-LSAT/main/data/AR_TestData.json", 
        Path("data/ar_lsat/original_test.json")
    )

def merge_with_reasoning_graph_data():
    print("Merging original AQUA-RAT with reasoning graph annotations...")
    aqua_rat_data = dataset_utils.AquaRatData()
    aqua_rat_data.merge_aqua_rat_data()
    print("Merging original AR-LSAT with reasoning graph annotations...")
    ar_lsat_data = dataset_utils.ARLSATData()
    ar_lsat_data.merge_ar_lsat_data()

def main():
    download_original_datasets()
    merge_with_reasoning_graph_data()

if __name__ == "__main__":
    main()
