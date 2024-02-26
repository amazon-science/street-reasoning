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
import json
import time
import logging
import argparse

from eval import evaluating
from utils import utils

from bleurt import score as bleurt_score

####################################################################
# Constants

ALL_SPLITS = ['train', 'dev', 'test']
ALL_STREET_TASKS = ['arc', 'scone', 'gsm8k', 'aqua_rat', 'ar_lsat']
ANNOTATED_DATA_PATH = 'data/{street_task_name}/reasoning_annotated_{split}.jsonl'

####################################################################
# Argument Parsing

def get_args():
    parser = argparse.ArgumentParser()    
    parser.add_argument(
        "--pred_file", default=None, required=True, type=str,
        help="Prediction file to score."
    )
    parser.add_argument(
        "--bleurt_checkpoint", default="", type=str,
        help="Path to the BLEURT model checkpoint. "
             "(Download from https://github.com/google-research/bleurt#checkpoints) "
             "We use bleurt-base-128 model for ARC and AR_LSAT evaluation."
    )
    parser.add_argument(
        "--split", default='test', type=str, 
        help="Which data split (train/dev/test) to evaluate."
    )
    parser.add_argument(
        "--street_task", default=None, type=str,
        help="Which street task (arc, scone, gsm8k, aqua_rat, ar_lsat) to evaluate. "
             "If not specified will test on all tasks."
    )
    

    args = parser.parse_args()
    return args

####################################################################
# Evaluation

def setup():
    json.encoder.FLOAT_REPR = lambda o: format(o, '.2f')

def get_logger():
    '''
    Creates a logger that writes to both stdout and  specified file.
    
    The DEBUG messages will only be written to the file. All other messages 
    will be written to stdout.
    '''    

    logger = logging.getLogger('__street__')
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        force=True
    )
    return logger

def initialize_bleurt_scorer(bleurt_scorer_model_path):
    '''
    Initializes BleurtScorer
    '''
    bleurt_scorer = bleurt_score.BleurtScorer(
        bleurt_scorer_model_path
    )
    return bleurt_scorer

def get_task_name_from_id(data_point_id):
    stree_task_name = re.match('([0-9A-Z\-]+)_.+', data_point_id).group(1)
    return stree_task_name.lower().replace('-', '_')

def get_evaluator_for_task(task_name, logger, bleurt_scorer):
    if task_name == 'arc':
        return evaluating.EntailmentARCEvaluator(
            logger=logger,
            bleurt_scorer = bleurt_scorer
        )
    if task_name == 'scone':
        return evaluating.EntailmentSconeEvaluator(logger=logger)
    if task_name == 'gsm8k':
        return evaluating.EntailmentGSM8KEvaluator(logger=logger)
    if task_name == 'aqua_rat':
        return evaluating.EntailmentAquaRatEvaluator(logger=logger)
    if task_name == 'ar_lsat':
        return evaluating.EntailmentArLsatEvaluator(
            logger=logger,
            bleurt_scorer = bleurt_scorer
        )

    raise ValueError(f'invalid task name {task_name}')

def run_evaluation(args):
    logger = get_logger()


    bleurt_scorer = None
    if args.bleurt_checkpoint != '':
        logger.info('\n' + '*' * 50)
        logger.info(f'INITIALIZING BLEURT SCORER')
        logger.info('*' * 50 + '\n')
        bleurt_scorer = initialize_bleurt_scorer(args.bleurt_checkpoint)
    pred_file = args.pred_file
    split = args.split

    assert split in ALL_SPLITS, f"ERROR: split {split} not valid."

    logger.info(f'\n** RUNNING STREET EVALUATION **\n')
    
    # load predicted file
    all_pred_data = utils.load_from_jsonl_file(pred_file)
    pred_data_map = {}

    for p_data_point in all_pred_data:
        pred_data_map[p_data_point['id']] = p_data_point

    # load golden data files
    for street_task_name in ALL_STREET_TASKS:

        if args.street_task and street_task_name != args.street_task.lower():
            continue

        logger.info('\n' + '*' * 50)
        logger.info(f'RUNNING EVALUATION FOR: "{street_task_name.upper()}"')
        logger.info('*' * 50 + '\n')

        evaluator = get_evaluator_for_task(
            street_task_name, logger, bleurt_scorer
        )
        task_golden_data_path = ANNOTATED_DATA_PATH.format(
            street_task_name = street_task_name,
            split = split,
        )
        task_golden_data = utils.load_from_jsonl_file(task_golden_data_path)
        task_pred_data = []
        for g_data_point in task_golden_data:
            # TODO: add feature to run on partial data.
            task_pred_data.append(
                pred_data_map[g_data_point['id']]
            )
        aggr_metrics = evaluator.compute_struct_metrics(
            task_golden_data, task_pred_data
        )
        logger.info('\n' + '*' * 50)
        logger.info(f'RESULTS FOR: "{street_task_name.upper()}"')
        logger.info(json.dumps(aggr_metrics, indent=2))        
        logger.info('*' * 50)

    logger.info('\n' + '*' * 50)
    logger.info('EVALUATION COMPLETE')
    logger.info('*' * 50)

def main():
    setup()
    args = get_args()    
    run_evaluation(args)

if __name__ == '__main__':
    main()