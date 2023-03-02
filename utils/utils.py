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

import os
import re
import sys
import csv
import copy
import json
import time
from pathlib import Path

####################################################################
# File Manipulation
####################################################################

def load_from_jsonl_file(path):
    with open(Path(path)) as fh:
        return [json.loads(line) for line in fh.readlines() if line]

def save_to_jsonl_file(output_list, file_path, verbose = True, 
                       make_dirs = True, open_mode = 'w'):
    if verbose:
        print('saving data to file:', file_path)
    if make_dirs:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir, exist_ok=True)
    with open(Path(file_path), open_mode) as file:
        for obj in output_list:
            file.write(json.dumps(obj) + '\n')
