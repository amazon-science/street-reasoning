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

from utils import utils

class ReasoningData():
    
    sent_part_separators = r'(\. )|(, )|(! )|(\? )|( and )|( then )'
    split_lst = ['train', 'dev', 'test']

    def break_question_into_parts(self, question):        
        question_parts = []
        last_part_idx = 0
        matches = list(re.finditer(self.sent_part_separators, question))
        for m_idx, match in enumerate(matches):
            match_text = match.group(0)            
            part = question[last_part_idx : match.end()].strip()
            if m_idx + 1 < len(matches):
                next_part = question[match.end():matches[m_idx+1].end()]
            else:
                next_part = question[match.end():]
            
            if (match_text.strip() in [',', 'and', 'then'] and 
                (len(part.split()) < 5 or len(next_part.split()) < 5)):
                # for certain split patterns, we avoid breaking sentences
                # where the parts would contains just a few tokens.
                continue
            
            question_parts.append(part)
            last_part_idx = match.end()
        
        if len(question[last_part_idx:]) > 2:
            question_parts.append(question[last_part_idx:].strip())
            
        return question_parts
    
    def format_question_for_prompting(self, question_parts):
        question_parts = '\n'.join([f'{idx+1}) {step}' 
                                    for idx, step in enumerate(question_parts)])
        question_formatted = f'Question:\n{question_parts}\nExplanation:\n'
        return question_formatted
        
    def get_answer_value(self, answer_text):
        pattern = 'The answer is (.+)\n'
        answer_value = re.search(pattern, answer_text, re.IGNORECASE)        
        if answer_value:
            answer_value = answer_value.group(1)
        return answer_value

    def load_graph_annotations(self):
        for split in self.split_lst:
            path = self.GRAPH_ANNOTATIONS_DATA_PATH.format(split = split)
            self.graph_annotations[split] = utils.load_from_jsonl_file(path)
    
    def init_annotated_datapoints(self):
        for split in self.split_lst:
            self.annotated_datapoints[split] = []

################################################################################
################################################################################
# AQUA-RAT (Algebra Question Answering with Rationales)
################################################################################
################################################################################


class AquaRatData(ReasoningData):
    '''
    Loads AQUA-RAT data
    
    Data example:
    {
        "question": "A grocery sells a bag of ice for $1.25, and makes 20% profit. If it sells 500 bags of ice, how much total profit does it make?",
        "options": ["A)125", "B)150", "C)225", "D)250", "E)275"],
        "rationale": "Profit per bag = 1.25 * 0.20 = 0.25\nTotal profit = 500 * 0.25 = 125\nAnswer is A.",
        "correct": "A"
    }
    '''

    ORIGINAL_DATA_PATH = 'data/aqua_rat/original_{split}.json'
    GRAPH_ANNOTATIONS_DATA_PATH = 'data/aqua_rat/aqua_rat_graph_annotation_{split}.jsonl'
    ANNOTATED_DATA_PATH = 'data/aqua_rat/reasoning_annotated_{split}.jsonl'
    
    def __init__(self):
        self.original_datapoints = {}
        self.graph_annotations = {}
        self.annotated_datapoints = {}
        self.load_original_dataset()
        self.load_graph_annotations()
        self.init_annotated_datapoints()

    def load_original_dataset(self):
        for split in self.split_lst:
            path = self.ORIGINAL_DATA_PATH.format(split = split)
            self.original_datapoints[split] = utils.load_from_jsonl_file(path)
            self.add_parts_to_examples(self.original_datapoints[split])

    def add_parts_to_examples(self, original_datapoints):
        for example in original_datapoints:
            example['question_parts'] = self.break_question_into_parts(
                example['question'])
            example['rationale_parts'] = self.break_rationale_into_parts(
                example['rationale'])

    def break_rationale_into_parts(self, rationale):
        # breaking rationales in from new lines should be good enough for GSM8K
        rationale_parts = [part for part in rationale.split('\n') if len(part) > 3]
        return rationale_parts
    
    def format_question_for_prompting(self, example):
        question_parts = example['question_parts']
        return super().format_question_for_prompting(question_parts)

    def merge_aqua_rat_data(self):
        for split in self.graph_annotations.keys():
            for graph_ann_dp_idx in range(len(self.graph_annotations[split])):
                graph_annotation = self.graph_annotations[split][graph_ann_dp_idx]
                original_dp_idx = int(
                    re.match(
                        "AQUA-RAT_(\d+)_[0-9a-z]+",
                        graph_annotation['id']
                    ).group(1)
                )
                original_datapoint = self.original_datapoints[split][original_dp_idx]

                # sanity check
                num_steps_1 = len(original_datapoint['rationale_parts'])
                num_steps_2 = len(graph_annotation['proof_edges'])
                assert num_steps_1 + 1 == num_steps_2

                all_consequent_nums = [
                    edge['consequent'] 
                    for edge in graph_annotation['proof_edges']
                ]
                first_conseq_num = min(all_consequent_nums)
                last_conseq_num = max(all_consequent_nums)

                all_tlus = (
                    original_datapoint['question_parts'] +
                    original_datapoint['options'] +
                    original_datapoint['rationale_parts'] + 
                    ['Answer is {answer}'.format(answer = original_datapoint['correct'])]
                )
                assert len(all_tlus) == last_conseq_num

                self.annotated_datapoints[split].append(
                    {
                        "id": graph_annotation['id'],
                        "context": "",
                        "question": original_datapoint['question'],                        
                        "options": original_datapoint['options'],
                        "rationale": original_datapoint['rationale'],
                        "answer": original_datapoint['correct'],
                        "textual_logical_units": {
                            int(tlu_idx): tlu
                            for tlu_idx, tlu in enumerate(all_tlus)
                        },
                        "reasoning_graph_edges": graph_annotation['proof_edges'],
                        "linearized_input": self.build_leaf_nodes(
                            question_parts = original_datapoint['question_parts'],
                            option_parts = original_datapoint['options']
                        ),
                        "linearized_output": self.build_inter_nodes(
                            question_parts = original_datapoint['question_parts'],
                            option_parts = original_datapoint['options'],
                            rationale_parts = original_datapoint['rationale_parts'],
                            proof_edges = graph_annotation['proof_edges'],
                            gold_answer = original_datapoint['correct']
                        ),
                        "metadata": [],
                    }
                )

            print(f"SPLIT = {split}")
            for i in range(3):
                print(json.dumps(self.annotated_datapoints[split][0], indent=2))
                print()
            utils.save_to_jsonl_file(
                output_list = self.annotated_datapoints[split],
                file_path = self.ANNOTATED_DATA_PATH.format(split = split)
            )

    @staticmethod
    def build_leaf_nodes(question_parts, option_parts):
        all_parts = question_parts + option_parts
        leaf_nodes = [f'sent{idx+1}: {all_parts[idx].strip()}' 
                      for idx in range(len(all_parts))]
        return ' '.join(leaf_nodes)

    @staticmethod
    def build_inter_nodes(question_parts, option_parts, rationale_parts, 
                          proof_edges, gold_answer):
        
        rationale_parts.append(f"The answer is {gold_answer})")
        premises_list = [
            (edge['antecedents'], edge['consequent'])
            for edge in proof_edges
        ]
        premises = [el[0] for el in sorted(premises_list, key=lambda x: x[1])]

        # sanity check
        assert len(rationale_parts) == len(premises)

        inter_nodes = []
        len_prem_parts = len(question_parts) + len(option_parts)
        last_step_idx = None
        for step_idx, (rationale_part, premise) in enumerate(zip(rationale_parts, premises)):
            prem_text_lst = []
            for p in premise:
                if p <= len_prem_parts:
                    prem_text_lst.append(f'sent{p}')
                else:
                    prem_text_lst.append(f'int{p-len_prem_parts}')
            assert len(prem_text_lst) > 0
            prem_text = ' & '.join(sorted(prem_text_lst))
            rationale_part = re.sub('[-]+>', ' => ', rationale_part)
            inter_nodes.append(f'{prem_text} -> int{step_idx+1}: {rationale_part};')
            last_step_idx = step_idx+1

        # make answer choice explicit and consistent formatting across all proofs
        # inter_nodes.append(f'int{last_step_idx} -> int{last_step_idx+1}: The answer is {gold_answer});')
        return ' '.join(inter_nodes)


################################################################################
################################################################################
# AR-LSAT (Analytical Reasoning LSAT)
################################################################################
################################################################################


class ARLSATData(ReasoningData):
    '''
    Loading AR-LSAT data and preparing it for annotation
    '''
    sent_part_separators = r'(\. )|(, )|(! )|(\? )|(—)|(: )|( and )|( then )'

    ORIGINAL_DATA_PATH = 'data/ar_lsat/original_{split}.json'
    GRAPH_ANNOTATIONS_DATA_PATH = 'data/ar_lsat/ar_lsat_graph_annotation_{split}.jsonl'
    ANNOTATED_DATA_PATH = 'data/ar_lsat/reasoning_annotated_{split}.jsonl'
    
    def __init__(self):
        self.original_datapoints = {}
        self.graph_annotations = {}
        self.annotated_datapoints = {}
        self.load_original_dataset()
        self.load_graph_annotations()
        self.init_annotated_datapoints()

    def load_original_dataset(self):
        for split in self.split_lst:
            filepath = self.ORIGINAL_DATA_PATH.format(
                split = split)
            split_data = utils.load_from_jsonl_file(filepath)
            split_data = split_data[0]
            self.original_datapoints[split] = []
            for datapoint in split_data:
                for question in datapoint['questions']:
                    example = {}
                    metadata = {}
                    example['passage'] = datapoint['passage']
                    example['answer_value'] = question['answer']
                    example['question'] = question['question']
                    example['options'] = question['options']
                    metadata['id'] = datapoint['id']
                    metadata['father_id'] = datapoint['fatherId']
                    metadata['passage_id'] = datapoint['passageId']
                    metadata['question_id'] = question['id']
                    metadata['question_father_id'] = question['fatherId']
                    metadata['question_id'] = question['questionId']
                    metadata['tags'] = list(filter(None, question['tags']))
                    example['metadata'] = metadata
                    self.original_datapoints[split].append(example)
            self.add_parts_to_examples(self.original_datapoints[split])
    
    def add_parts_to_examples(self, original_datapoints):
        for example in original_datapoints:
            example['context_parts'] = self.break_question_into_parts(
                example['passage'])
            example['question_parts'] = self.break_question_into_parts(
                example['question'])
            example['options_parts'] = self.break_options_into_parts(
                example['options'])

    def break_options_into_parts(self, options):
        ret_options = []
        for o_idx, opt in enumerate(options):
            option_letter = chr(ord('A') + o_idx)
            ret_options.append(f'{option_letter}) {opt}')
        return ret_options

    def merge_ar_lsat_data(self):
        for split in self.graph_annotations.keys():
            for graph_ann_dp_idx in range(len(self.graph_annotations[split])):
                graph_annotation = self.graph_annotations[split][graph_ann_dp_idx]
                original_dp_idx = int(re.match(
                    "AR-LSAT_(\d+)_([0-9a-zA-Z_-]+)",
                    graph_annotation['id']
                ).group(1))

                original_datapoint = self.original_datapoints[split][original_dp_idx]
                context_parts = original_datapoint['context_parts']
                question_parts = original_datapoint['question_parts']
                options_parts = original_datapoint['options_parts']
                proof_edges = graph_annotation['proof_edges']
                all_consequent_nums = [
                    edge['consequent'] 
                    for edge in proof_edges
                ]
                first_conseq_num = min(all_consequent_nums)
                last_conseq_num = max(all_consequent_nums)
                
                num_leaf_nodes = (
                    len(context_parts) +
                    len(question_parts) + 
                    len(options_parts)
                )
                # sanity checks
                num_steps_1 = len(graph_annotation['rationale_parts'])
                num_steps_2 = len(proof_edges)
                assert num_steps_1 == num_steps_2
                if first_conseq_num != num_leaf_nodes + 1:
                    print(f"split = {split}\n")
                    print(f"graph_ann_dp_idx = {graph_ann_dp_idx}\n")
                    print(f"original_dp_idx = {original_dp_idx}\n")
                    print(f"first_conseq_num = {first_conseq_num}\n")
                    print(f"num_leaf_nodes = {num_leaf_nodes}\n")
                    print(f"proof_edges = {proof_edges}\n")
                    print(f"context_parts = {context_parts}\n")
                    print(f"question_parts = {question_parts}\n")
                    print(f"options_parts = {options_parts}\n")
                    print()
                assert first_conseq_num == num_leaf_nodes + 1

                # map from rationale number and rationale text                
                explanation_dict = {
                    first_conseq_num + r_idx: rationale
                    for r_idx, rationale in enumerate(
                        graph_annotation['rationale_parts']
                    )
                }
                # map from rationale number to its premises
                premises_dict = {
                    edge['consequent']: edge['antecedents']
                    for edge in proof_edges
                }
                all_tlus = (
                    original_datapoint['context_parts'] + 
                    original_datapoint['question_parts'] +
                    original_datapoint['options_parts'] +                    
                    graph_annotation['rationale_parts']
                )
                assert len(all_tlus) == last_conseq_num

                leaf_nodes = self.build_leaf_nodes(
                    context_parts = context_parts,
                    question_parts = question_parts,
                    options_parts = options_parts
                )
                inter_nodes = self.build_inter_nodes(
                    context_parts = context_parts,
                    question_parts = question_parts,
                    options_parts = options_parts,
                    explanation_dict = explanation_dict, 
                    premises_dict = premises_dict, 
                    gold_answer = original_datapoint['answer_value']
                )

                self.annotated_datapoints[split].append(
                    {
                        "id": graph_annotation['id'],
                        "context": original_datapoint['passage'],
                        "question": original_datapoint['question'],                        
                        "options": original_datapoint['options'],
                        "rationale": ' '.join(graph_annotation['rationale_parts']),
                        "answer": original_datapoint['answer_value'],
                        "textual_logical_units": {
                            int(tlu_idx): tlu
                            for tlu_idx, tlu in enumerate(all_tlus)
                        },
                        "reasoning_graph_edges": graph_annotation['proof_edges'],
                        "linearized_input": leaf_nodes,
                        "linearized_output": inter_nodes,
                        "metadata": [],
                    }
                )

            print(f"SPLIT = {split}")
            for i in range(3):
                print(json.dumps(self.annotated_datapoints[split][0], indent=2))
                print()
            utils.save_to_jsonl_file(
                output_list = self.annotated_datapoints[split],
                file_path = self.ANNOTATED_DATA_PATH.format(split = split)
            )

    @staticmethod
    def build_leaf_nodes(context_parts, question_parts, options_parts):
        all_parts = context_parts + question_parts + options_parts
        leaf_nodes = [f'sent{idx+1}: {all_parts[idx].strip()}' 
                      for idx in range(len(all_parts))]
        return ' '.join(leaf_nodes)

    @staticmethod
    def build_inter_nodes(context_parts, question_parts, options_parts, explanation_dict, 
                          premises_dict, gold_answer):        
        inter_nodes = []
        len_prem_parts = len(context_parts) + len(question_parts) + len(options_parts)
        last_step_idx = None
        
        # Fix bug when explanation and premise numbers are not 
        # consecutive from options and questions
        explan_offset = min(explanation_dict.keys()) - len_prem_parts - 1
        
        for step_idx, (tlu_num, rationale_part) in enumerate(explanation_dict.items()):
            premise = []
            if tlu_num in premises_dict:
                premise = premises_dict[tlu_num]
            prem_text_lst = []
            for p in premise:
                if p <= len_prem_parts:
                    prem_text_lst.append(f'sent{p}')
                else:
                    prem_text_lst.append(f'int{p-len_prem_parts-explan_offset}')
            if len(prem_text_lst) == 0:
                # use sent0 as a "null" premise, this way I won't have to 
                # worry about the formatting of the code.
                prem_text_lst.append('sent0')
            assert len(prem_text_lst) > 0
            prem_text = ' & '.join(sorted(prem_text_lst))
            rationale_part = re.sub('[-]+>', ' => ', rationale_part)
            inter_nodes.append(f'{prem_text} -> int{step_idx+1}: {rationale_part};')
            last_step_idx = step_idx+1        
        return ' '.join(inter_nodes)
