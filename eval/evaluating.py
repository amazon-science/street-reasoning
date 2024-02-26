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

import re
import copy
import collections

from tqdm import tqdm

from eval import scoring
from eval import parsing

class EntailmentStructuredExpEvaluator:
    '''
    Evaluator for most experiments in the "STREET" benchmark paper
    '''

    def __init__(self, logger = None, bleurt_scorer = None):
        self.logger = logger
        self.bleurt_scorer = bleurt_scorer
    
    def compute_struct_metrics(self, datapoints, pred_results, verbose=False):
        predictions_tot = 0.0
        proof_correct_tot = 0.0
        answer_correct_tot = 0.0
        premise_correct_tot = 0.0
        graph_sim_tot = 0.0
        
        assert len(datapoints) == len(pred_results)
        for dp_it in tqdm(range(len(datapoints))):
            datapoint =  datapoints[dp_it]
            pred_datapoint = pred_results[dp_it]
            
            predictions_tot += 1.0

            pred_proof = pred_datapoint['linearized_output']
            first_pos = re.search('sent|int', pred_proof)
            if first_pos is not None:
                pred_proof = pred_proof[first_pos.start():]
            
            # compare proofs
            target_proof = datapoint['linearized_output']
            target_answer = self.get_answer_value(
                datapoint, target_proof, verbose=verbose
            )
            pred_answer = self.get_answer_value(
                datapoint, pred_proof, verbose=verbose
            )
            
            is_correct_answer = target_answer == pred_answer
            answer_correct_tot += 1.0 if is_correct_answer else 0.0
            
            broken_proof = False
            try:
                _ = parsing.parse_reasoning_proof(
                    pred_proof, ignore_broken_step=False, 
                    enforce_nodes_in_contex = False
                )
            except:
                broken_proof = True
                # broken proof is an incorrect proof
                # if dp_it < 10 and verbose:
                if dp_it < 10 and verbose:
                    self.logger.info(f'broken proof example = {pred_proof}')
            
            # test if proofs are equivalent (will perform a matching, 
            # where leaf nodes are the same
            is_proof_equivalent = False
            is_proof_equivalent_ignore_inter = False
            graph_sim = 0.0
            if not broken_proof:                            

                is_proof_equivalent = scoring.proof_is_equivalent(
                    pred_proof, target_proof,                    
                    similarity_fn = self.proof_equivalent_similarity_fn)
                if is_proof_equivalent:
                    proof_correct_tot += 1.0                

                is_proof_equivalent_ignore_inter = scoring.proof_is_equivalent(
                    pred_proof, target_proof, 
                    similarity_fn = None)
                if (is_proof_equivalent_ignore_inter and 
                    is_correct_answer):
                    premise_correct_tot += 1.0
                if is_correct_answer:
                    # only compute graph similarity if answer is correct
                    graph_sim = scoring.proof_edit_distance(
                        pred_proof, target_proof,
                        similarity_fn = self.proof_equivalent_similarity_fn,
                        return_similarity = True
                    )
                    graph_sim_tot += graph_sim

            if dp_it < 50 and verbose:
                if is_correct_answer:
                    self.logger.info('YES CORRECT ANSWER!')
                else:
                    self.logger.info('NOT CORRECT ANSWER!')
                if is_proof_equivalent:
                    self.logger.info('YES CORRECT PROOF')
                else:
                    self.logger.info('NOT CORRECT PROOF')             
                if is_proof_equivalent_ignore_inter:
                    self.logger.info('YES PREMISE SELECTION CORRECT')
                else:
                    self.logger.info('NOT PREMISE SELECTION CORRECT')
                self.logger.info('context =')
                self.logger.info(datapoint['linearized_input'])
                self.logger.info('target_proof =')
                self.logger.info(target_proof.replace('; ', ';\n'))
                self.logger.info('pred_proof =') 
                self.logger.info(pred_proof.replace('; ', ';\n'))
                self.logger.info('target_answer =')
                self.logger.info(str(target_answer))
                self.logger.info('pred_answer =')
                self.logger.info(str(pred_answer))
                self.logger.info(f'graph_sim = {graph_sim}\n')
        
        assert int(predictions_tot) == len(pred_results)
            
        aggr_metrics = {            
            'answer-acc': answer_correct_tot / predictions_tot,
            'reasoning-graph-acc': proof_correct_tot / predictions_tot,
            'premise-selection-acc': premise_correct_tot / predictions_tot,
            'graph-similarity': graph_sim_tot / predictions_tot,
        }
        return aggr_metrics

    def is_conclusion_text_bleurt_similarity_correct(
        self, candidate, reference, score_threshold = 0.25
    ):
        scores = self.bleurt_scorer.score(
            references=[reference], candidates=[candidate]
        )
        
        assert type(scores) == list and len(scores) == 1        
        # we will allow for more diverse set
        return scores[0] > score_threshold

################################################################################
################################################################################
# ARC 
################################################################################
################################################################################

class EntailmentARCEvaluator(EntailmentStructuredExpEvaluator):

    def __init__(self, *args, **kwargs):        
        super().__init__(*args, **kwargs)        
    
    def proof_equivalent_similarity_fn(self, text1 , text2):        
        return self.is_conclusion_text_similarity_correct(candidate = text1, reference = text2)
    
    def get_answer_value(self, datapoint, proof, verbose=False):
        match = re.search(r'(The answer is )([ABCDEF]\))', proof, re.IGNORECASE)
        if not match:
            # use default answer
            return 'A)'
        answer_value = match.group(2)
        return answer_value
    
    def is_conclusion_text_similarity_correct(self, candidate, reference):
        is_similar = self.is_conclusion_text_bleurt_similarity_correct(
            reference = reference, candidate = candidate
        )
        return is_similar
    
################################################################################
################################################################################
# SCONE
################################################################################
################################################################################


class SconeProcessor():
    
    task_lst = ['alchemy', 'scene', 'tangrams']
    split_lst = ['train', 'dev', 'test']
    to_color_name_map = {
        'r': 'red', 'y': 'yellow',
        'g': 'green', 'o': 'orange', 'p': 'purple', 
        'b': 'brown'
    }
    
    def __init__(self):
        self.to_color_symb_map = {v: k for k,v in self.to_color_name_map.items()}
    
    def to_leaf_node_symb(self, pos):
        return f'sent{pos}'

    def leaf_node_symb_num(self, leaf_node_symb):
        return int(leaf_node_symb[len('sent'):])

    
class SconeAlchemyProcessor(SconeProcessor):
    '''
    Data processor to parse proof from SCONE ALCHEMY
    '''
    
    alchemy_num_beakers = 7
    alchemy_num_actions = 5
    ordinal_numbers = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth']
    
    def __init__(self):
        super().__init__()
        self.ordinal_numbers_to_int = {
            el: (idx+1) for idx, el in enumerate(self.ordinal_numbers)}
        self.to_chemical_symb_map = dict(self.to_color_symb_map)
        
    def from_node_text_to_beaker_state(self, node_text):
        '''
        Text pattern for Alchemy is:

        "$beaker position$ beaker has $quant of chem$ $color of chem$ 
        [and $quant of chem$ $color of chem$]* chemical[s]"
        '''
        tokens = node_text.split()
        beaker_pos = self.ordinal_numbers_to_int[tokens[0]]
        tokens = tokens[3:] # ditch "$beaker position$ beaker has"
        chems = []
        while len(tokens) > 0:            
            quantity = int(tokens[0])
            if quantity == 0:
                # corner case when beaker has no chemicals
                return f'{beaker_pos}:_'
            chem_symb = self.to_chemical_symb_map[tokens[1]]
            chems.append((quantity, chem_symb))
            tokens = tokens[3 if tokens[2] == 'and' else 2:]            
            if tokens[0].startswith('che'):
                break
        chem_symbs = ''.join([syb * num  for (num, syb) in chems])
        return f'{beaker_pos}:{chem_symbs}'


    def get_alchemy_states_from_proof(self, leaf_nodes, inter_nodes):
        '''
        list of alquemy states (list of strings representaing states), 
        each state will be  in the  format like "1:oo 2:_ 3:_ ..."
        '''
        states_lst = []
        current_state = ['' for i in range(self.alchemy_num_beakers)]
        for leaf_node in leaf_nodes[:self.alchemy_num_beakers]:
            leaf_text = leaf_node.split(': ')[1]
            beaker_state = self.from_node_text_to_beaker_state(leaf_text)
            beaker_pos = int(beaker_state[0])
            current_state[beaker_pos-1] = beaker_state

        states_lst.append(copy.deepcopy(current_state))
        action_range = range(self.alchemy_num_beakers, 
                             self.alchemy_num_beakers + self.alchemy_num_actions)
        all_action_leaves = set([f'sent{num+1}' for num in action_range])
        last_action_leaf = None
        for inter_node in inter_nodes:
            inter_text = inter_node.split(': ')[1]
            beaker_state = self.from_node_text_to_beaker_state(inter_text)
            beaker_pos = int(beaker_state[0])            
            current_state[beaker_pos-1] = beaker_state
            action_leaf = [el for el in inter_node.split() if el in all_action_leaves]
            assert len(action_leaf) == 1
            action_leaf = action_leaf[0]            
            if last_action_leaf and action_leaf == last_action_leaf:
                # update state since it is the same action
                states_lst[-1] = copy.deepcopy(current_state)
            else:
                states_lst.append(copy.deepcopy(current_state))
            last_action_leaf = action_leaf

        assert len(states_lst) == self.alchemy_num_actions + 1
        return states_lst
        
class SconeSceneProcessor(SconeProcessor):
    '''
    Data processor to parse proof from SCONE SCENE Task
    '''
    
    scene_num_positions = 10
    scene_num_actions = 5
    
    def __init__(self):
        super().__init__()
        self.shirt_to_name = dict(self.to_color_name_map)
        self.hat_to_name = dict(self.to_color_name_map)
        self.hat_to_name.update({'e': 'no'})

    def from_node_text_to_scene_state(self, node_text):
        '''
        Text pattern for SCENE is:

        "position $int:position$ has person in $str:shirt_color$ shirt and [$str:hat_color$|no] hat"
        '''
        tokens = node_text.split()
        position = int(tokens[1])

        if tokens[3] == 'no':
            return f'{position}:__'
        shirt, hat = tokens[5], tokens[8]
        shirt = self.to_color_symb_map[shirt]
        if hat == 'no':
            hat = 'e'
        else:
            hat = self.to_color_symb_map[hat]
        return f'{position}:{shirt}{hat}'

    def get_scene_states_from_proof(self, leaf_nodes, inter_nodes):
        '''
        list of SCENE states (list of strings representaing states), 
        each state will be  in the  format like "1:pr 2:__ 3:re ..."
        '''
        states_lst = []
        current_state = ['' for i in range(self.scene_num_positions)]

        # create initial state from leaf nodes
        for leaf_node in leaf_nodes[:self.scene_num_positions]:
            leaf_text = leaf_node.split(': ')[1]
            person_state = self.from_node_text_to_scene_state(leaf_text)
            p_pos = int(person_state.split(':')[0])
            current_state[p_pos-1] = person_state

        states_lst.append(copy.deepcopy(current_state))
        action_range = range(self.scene_num_positions, 
                             self.scene_num_positions + self.scene_num_actions)
        all_action_leaves = set([f'sent{num+1}' for num in action_range])
        last_action_leaf = None
        for inter_node in inter_nodes:        
            action_leaf = [el for el in inter_node.split() if el in all_action_leaves]
            assert len(action_leaf) == 1
            action_leaf = action_leaf[0]            

            # handle actions that are no op (has no inter_node), (e.g. "nobody moves")
            for _ in range(len(states_lst), 
                      self.leaf_node_symb_num(action_leaf) - self.scene_num_positions):
                states_lst.append(copy.deepcopy(current_state))

            inter_text = inter_node.split(': ')[1]
            person_state = self.from_node_text_to_scene_state(inter_text)
            per_pos = int(person_state.split(':')[0])
            
            current_state[per_pos-1] = person_state

            if last_action_leaf and action_leaf == last_action_leaf:
                # update state since it is the same action
                states_lst[-1] = copy.deepcopy(current_state)
            else:
                states_lst.append(copy.deepcopy(current_state))

            last_action_leaf = action_leaf

        # handle actions that are no op (has no inter_node), (e.g. "nobody moves")
        # this is only for the last action step.
        if self.leaf_node_symb_num(last_action_leaf) - self.scene_num_positions < self.scene_num_actions:            
            for _ in range(self.scene_num_actions - self.leaf_node_symb_num(last_action_leaf) 
                           + self.scene_num_positions):
                states_lst.append(copy.deepcopy(current_state))

        assert len(states_lst) == self.scene_num_actions + 1
        return states_lst

        
class SconeTangramsProcessor(SconeProcessor):
    '''
    Data processor to parse proof from SCONE TANGRAMS Task
    '''
    
    tangrams_num_positions = 5
    tangrams_num_actions = 5
    
    def __init__(self):
        super().__init__()
        self.tangrams_shape_to_name = {
            i: chr(ord('A') + i) for i in range(self.tangrams_num_positions)}
        self.name_to_tangrams_shape = {
            v: k for k, v in self.tangrams_shape_to_name.items()}

    def from_node_text_to_tangrams_state(self, node_text):
        '''
        Text pattern for TANGRAMS is:

        "position $int:position$ has figure $str:shape$"
        '''
        tokens = node_text.split()
        position = int(tokens[1])

        # e.g. "position 3 has no figure"
        if tokens[3] == 'no':
            return f'{position}:_'
        shape = tokens[4]
        shape_id = self.name_to_tangrams_shape[shape]
        return f'{position}:{shape_id}'

    def get_tangrams_states_from_proof(self, leaf_nodes, inter_nodes):
        '''
        list of TANGRAMS states (list of strings representaing states), 
        each state will be  in the  format like "1:1 2:2 3:_ ..."
        '''
        states_lst = []
        current_state = ['' for i in range(self.tangrams_num_positions)]

        # create initial state from leaf nodes
        for leaf_node in leaf_nodes[:self.tangrams_num_positions]:
            leaf_text = leaf_node.split(': ')[1]
            person_state = self.from_node_text_to_tangrams_state(leaf_text)
            p_pos = int(person_state.split(':')[0])
            current_state[p_pos-1] = person_state

        states_lst.append(copy.deepcopy(current_state))
        action_range = range(self.tangrams_num_positions, 
                             self.tangrams_num_positions + self.tangrams_num_actions)
        all_action_leaves = set([f'sent{num+1}' for num in action_range])
        last_action_leaf = None
        for inter_node in inter_nodes:
            inter_text = inter_node.split(': ')[1]
            person_state = self.from_node_text_to_tangrams_state(inter_text)
            per_pos = int(person_state.split(':')[0])
            
            current_state[per_pos-1] = person_state
            action_leaf = [el for el in inter_node.split() if el in all_action_leaves]
            assert len(action_leaf) == 1
            action_leaf = action_leaf[0]
            
            if last_action_leaf and action_leaf == last_action_leaf:
                # update state since it is the same action
                states_lst[-1] = copy.deepcopy(current_state)
            else:
                states_lst.append(copy.deepcopy(current_state))
            last_action_leaf = action_leaf

        assert len(states_lst) == self.tangrams_num_actions + 1
        return states_lst


class EntailmentSconeEvaluator(EntailmentStructuredExpEvaluator):
    
    def __init__(self, *args, **kwargs):        
        super().__init__(*args, **kwargs)        
        self.alchemy_processor = SconeAlchemyProcessor()
        self.scene_processor = SconeSceneProcessor()
        self.tangrams_processor = SconeTangramsProcessor()

    def proof_equivalent_similarity_fn(self, text1 , text2):
        # scone uses strict textual similarity since it is a "generated" dataset
        return text1 == text2
    
    def get_answer_value(self, datapoint, proof, verbose=False):
        inter_nodes = proof.split(';')[:-1]
        parsed_context = parsing.parse_reasoning_context(
            datapoint['linearized_input']
        )
        leaf_nodes = [f'{k}: {v["text"]}' for k, v  in parsed_context.items()]
        
        scone_states = None
        try:
            if 'ALCHEMY' in datapoint['id']:
                scone_states = self.alchemy_processor.get_alchemy_states_from_proof(
                    leaf_nodes = leaf_nodes, inter_nodes = inter_nodes
                )
            elif 'SCENE' in datapoint['id']:
                scone_states = self.scene_processor.get_scene_states_from_proof(
                    leaf_nodes = leaf_nodes, inter_nodes = inter_nodes
                )
            elif 'TANGRAMS' in datapoint['id']:
                scone_states = self.tangrams_processor.get_tangrams_states_from_proof(
                    leaf_nodes = leaf_nodes, inter_nodes = inter_nodes
                )
            else:
                raise NotImplementedError('SCONE Task not valid')
        except Exception as e:
            if verbose:
                self.logger.info('EXCEPTION WHEN PARSING SCONE STATE')
                self.logger.info(str(e))
                self.logger.info(f'leaf_nodes = {leaf_nodes}')
                self.logger.info(f'inter_nodes = {inter_nodes}')            

        # use last state string as answer 
        # state is formated as '1:[state 1] 2:[state 2] 3:...'
        
        if scone_states:
            answer_value = scone_states[-1]
        else:
            # use full proof as default answer
            answer_value = proof
        return answer_value

################################################################################
################################################################################
# GSM8K
################################################################################
################################################################################    
    
class EntailmentGSM8KEvaluator(EntailmentStructuredExpEvaluator):
    
    def __init__(self, *args, **kwargs):        
        super().__init__(*args, **kwargs)
    
    def proof_equivalent_similarity_fn(self, text1 , text2):
        return self.test_number_only_similarity_fn(text1 , text2)
    
    def test_number_only_similarity_fn(self, text1 , text2):
        '''
        For now two nodes are similar if they contain the same numbers
        
        NOTE: this function disregards the order that the numbers appear in the text
        '''        
        numbers_1 = collections.Counter(list(re.findall(r'\d+', text1)))
        numbers_2 = collections.Counter(list(re.findall(r'\d+', text2)))        
        return numbers_1 == numbers_2
    
    def get_answer_value(self, datapoint, proof, verbose=False):
        match = re.search(r'(The answer is )(.+);', proof, re.IGNORECASE)
        if not match:
            # use default answer 
            return 1
        answer = match.group(2)
        # select the first number only, disregard other tokens
        answer_num_lst = list(re.findall(r'\d+', answer))
        if len(answer_num_lst) > 0:
            answer = answer_num_lst[0]
        return answer
        
################################################################################
################################################################################
# AQUA-RAT
################################################################################
################################################################################

class EntailmentAquaRatEvaluator(EntailmentStructuredExpEvaluator):
    
    def __init__(self, *args, **kwargs):        
        super().__init__(*args, **kwargs)
    
    def proof_equivalent_similarity_fn(self, text1 , text2):
        return self.test_number_or_choice_only_similarity_fn(text1 , text2)
    
    def test_number_or_choice_only_similarity_fn(self, text1 , text2):
        '''
        For now two nodes are similar if they contain the same numbers
        
        NOTE: this function disregards the order that the numbers appear in the text
        '''        
        vals_1 = collections.Counter(list(re.findall(
            r'(\d+)|(The answer is [ABCDEF]\))', text1)))
        vals_2 = collections.Counter(list(re.findall(
            r'(\d+)|(The answer is [ABCDEF]\))', text2)))
        return vals_1 == vals_2
    
    def get_answer_value(self, datapoint, proof, verbose=False):
        match = re.search(r'(The answer is )([ABCDEF]\))', proof, re.IGNORECASE)
        if not match:
            # use default answer
            return 'A)'
        answer_value = match.group(2)
        return answer_value
    
################################################################################
################################################################################
# AR-LSAT
################################################################################
################################################################################

class EntailmentArLsatEvaluator(EntailmentStructuredExpEvaluator):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        
    
    def proof_equivalent_similarity_fn(self, text1 , text2):
        return self.is_conclusion_text_similarity_correct(candidate = text1, reference = text2)
    
    def get_answer_value(self, datapoint, proof, verbose=False):
        match = re.search(r'(The answer is )([ABCDEF]\))', proof, re.IGNORECASE)
        if not match:
            # use default answer
            return 'A)'
        answer_value = match.group(2)
        return answer_value


    def is_conclusion_text_similarity_correct(self, candidate, reference):
        is_similar = self.is_conclusion_text_bleurt_similarity_correct(
            reference = reference, candidate = candidate
        )
        return is_similar