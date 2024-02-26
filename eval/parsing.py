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

####################################################################
# Reasoning graph parsing
####################################################################

def is_leaf_node_symbol(node_symbol):
    '''
    Returns true if node symbol is in the format 'sentX'
    '''
    return re.fullmatch('sent[0-9]+', node_symbol) is not None

def is_intermediate_node_symbol(node_symbol):
    '''
    Returns true if node symbol is in the format 'sentX'
    '''
    return re.fullmatch('int[0-9]+', node_symbol) is not None

def get_node_symbol_number(node_symbol):
    '''
    Returns the number of the symbol (e.g. sent1 return 1, int2 return 2)
    If not a node symbol return None
    '''
    symbol_number = re.fullmatch('(int|sent)([0-9]+)', node_symbol)
    if symbol_number:
        symbol_number = symbol_number.group(2)
        if not symbol_number.isdigit():
            return None
        return int(symbol_number)
    return None
    
def proof_step_follows_dag_structure(last_proof_step, prev_proof_steps):
    '''
    Takes the last proof step and the list of previous proof steps in the format:
    [{
        'antecedents': [{'symbol': ant_symbol}],
        'consequent': {
            'symbol': conseq_symb,
        }
    }]
    And returns true if and only if it follows the following constraints
    
    2 - check that consequent has not been used in previous steps
    3 - check that consequent symbol number is in incremental order
    4 - check theat there are no self loops in consequent
    '''    
    inter_symbol = last_proof_step['consequent']['symbol']
    
    # compute used nodes from previous steps
    used_consequent = set()
    for step in prev_proof_steps:
        used_consequent.add(step['consequent']['symbol'])
    
    # check values of last step
    for antecedent in last_proof_step['antecedents']:
        ant_symbol = antecedent['symbol']
        if (not (is_leaf_node_symbol(ant_symbol) or 
                 is_intermediate_node_symbol(ant_symbol))):            
            # in a DAG non-root nodes have to be of the format sentX or intX            
            return False
        if ant_symbol == inter_symbol:
            # No self-loops allowed in DAGs
            return False
    
    prev_inter_symbol = None
    if len(prev_proof_steps) > 0 and inter_symbol != 'hypothesis':
        prev_inter_symbol = prev_proof_steps[-1]['consequent']['symbol']
        inter_num = get_node_symbol_number(inter_symbol)
        prev_inter_num = get_node_symbol_number(prev_inter_symbol)
        if (not inter_num or not prev_inter_num or 
            inter_num - prev_inter_num != 1):            
            # every step intermediate node should follow increasing number in order
            return False

    if (inter_symbol in used_consequent or
        not (is_intermediate_node_symbol(inter_symbol) or 
             inter_symbol == 'hypothesis')):        
        # every step should represent a different tree node        
        return False
    return True

def proof_step_follows_tree_structure(last_proof_step, prev_proof_steps):
    '''
    Takes the last proof step and the list of previous proof steps in the format:
    [{
        'antecedents': [{'symbol': ant_symbol}],
        'consequent': {
            'symbol': conseq_symb,
        }
    }]
    And returns true if and only if it follows the following constraints
    
    1 - check that antecedents have only been used once
    2 - check that consequent has not been used in previous steps
    3 - check that consequent symbol number is in incremental order
    4 - check theat there are no self loops in consequent
    
    Last three are DAG constraints
    '''
    
    # compute used nodes from previous steps
    used_antecedent = set()
    for step in prev_proof_steps:
        for antecedent in step['antecedents']:
            ant_symbol = antecedent['symbol']
            used_antecedent.add(ant_symbol)
    
    # check values of last step
    for antecedent in last_proof_step['antecedents']:
        ant_symbol = antecedent['symbol']
        if ant_symbol in used_antecedent:
            # in a tree every node can only have one parent
            # print('in a tree every node can only have one parent')
            return False
    
    # same checks as DAG
    return proof_step_follows_dag_structure(last_proof_step, prev_proof_steps)

def proof_step_nodes_in_context(proof_step, symbols_in_context):
    '''
    Takes the a proof step in the format:
    {
        'antecedents': [{'symbol': ant_symbol}, ...],
        'consequent': {
            'symbol': conseq_symb,
        }
    }
    And returns True if and only if all node symbols are in 
    the input list "symbols_in_context"
    '''
    for node in proof_step['antecedents']:
        node_symbol = node['symbol']
        if node_symbol == 'sent0':
            # this is a special case for datasets that contain reasoning steps
            # without antecedents
            continue
        if node_symbol not in symbols_in_context:
            # print(f'node_symbol not in context: {node_symbol}')
            return False
        
    return True

def finished_proof_uses_all_context(last_proof_step, prev_proof_steps, symbols_in_context):
    '''
    Takes the list of proof steps in the format:
    [{
        'antecedents': [{'symbol': ant_symbol}],
        'consequent': {
            'symbol': conseq_symb,
        }
    }]
    And returns True if and only if proof if not finished (step does not contains 
    hypothesis as conclusion) or all leaf nodes are in used in some step.
    '''
    if last_proof_step['consequent']['symbol'] != 'hypothesis':
        # proof unfinished
        return True
    all_steps = [last_proof_step] + prev_proof_steps
    used_leaf_nodes = set()
    for step in all_steps:
        for ant in step['antecedents']:            
            used_leaf_nodes.add(ant['symbol'])
    for context_symb in symbols_in_context:
        if (is_leaf_node_symbol(context_symb) and 
            context_symb not in used_leaf_nodes):
            # symbol in context not used
            return False            
    return True

def parse_reasoning_context(context):
    '''
    Transform a context in the form "sent1: [...] sent2: [...]", breaking 
    down context into dictionary of premsies that maps symbols to text.    
    '''
    parsed_context = {}
    sent_matches = list(re.finditer("(sent)[0-9]+:", context))    
    for match_idx, match in enumerate(sent_matches):
        sent_match = match.group()
        sent_symb = sent_match[:-1] # remove the ':' in "sentX:'
        sent_span = match.span()
        start_pos = sent_span[0] + len(sent_match)
        end_pos = None
        if match_idx + 1 < len(sent_matches):
            end_pos = sent_matches[match_idx + 1].span()[0]
        sent_text = context[start_pos: end_pos].strip()
        parsed_context[sent_symb] = {
            'text': sent_text
        }
    return parsed_context


def parse_reasoning_step(step, symbol_to_text = {},
                        enforce_multi_antecedent = False, 
                        return_none_if_error = False):
    '''
    Breaks down a step in the text format (e.g. "sent2 & sent4 & int2 -> int2: earth is aplanet[;]")
    
    Every node in the step will be represented by a dictionary as in: {'symbol': node_symbol, 'text': node_text}
    where node_text is extracted from input symbol_to_text (if provided)
    
    Will raise an ValueError exception if can't parse step.
    
    - symbol_to_text: a map between node symbols (e.g. "sent1" or "int2" to their 
        textual representation), will update this input with consequent text if provided
    - enforce_multi_antecedent: if True will throw an error for steps with one
        single antecendet
    - return_none_if_error: returns (None, symbol_to_text) instead of raising exception if parse fails
    '''
    if len(step) == 0:
        if return_none_if_error:
            return None, symbol_to_text
        raise ValueError(f'Step is empty! {step}')
    if step[-1] == ';':
        step = step[:-1]
    if step.count('->') != 1:
        if return_none_if_error:
            return None, symbol_to_text
        raise ValueError(f'No consequent in step! {step}')
    antecedents_str, consequent_str = [el.strip() for el in step.split('->')]        
    conseq_symb = 'hypothesis'
    # uses empty text if hypothesis not provided
    conseq_text = symbol_to_text[conseq_symb] if conseq_symb in symbol_to_text else None
    if consequent_str != 'hypothesis':
        # consequent is of the form "int[0-9]+: [...]"
        if len(list(re.findall('int[0-9]+: ', consequent_str))) != 1:
            if return_none_if_error:
                return None, symbol_to_text
            raise ValueError(f'Consequent not properly formated! {step}')
        split_pos = re.match('int[0-9]+: ', consequent_str).span()[1]
        # disregard the chars ": " with -2
        conseq_symb = consequent_str[:split_pos-2].strip()
        conseq_text = consequent_str[split_pos:].strip()
        # add consequent text to context
        symbol_to_text[conseq_symb] = conseq_text

    if antecedents_str.count('&') == 0 and enforce_multi_antecedent:
        if return_none_if_error:
            return None, symbol_to_text
        raise ValueError(f'Not enough antecedents in step! {step}')

    antecedents_symbols = [ant.strip() for ant in antecedents_str.split('&')]
    antecedents = [
        {
            'symbol': ant, 
            # uses empty text if context not provided
            'text': symbol_to_text[ant] if ant in symbol_to_text else None
        }
        for ant in antecedents_symbols
    ]
    
    parsed_step = {
        'antecedents': antecedents,
        'consequent': {
            'symbol': conseq_symb,
            'text': conseq_text
        }
    }
    return parsed_step, symbol_to_text

def parse_reasoning_proof(proof, hypothesis = None, context = None, 
                         ignore_broken_step = False,
                         enforce_dag_structure = True,
                         enforce_nodes_in_contex = True,                         
                         enforce_finished_uses_all_contex = False,                         
                         enforce_multi_antecedent = False,
                         enforce_tree_structure = False):
    '''
    Transform a proof in the form "A & B -> C; C & D -> H;", breaking 
    down proof into list of steps, leaves, intermediate nodes and hypothesis.
    
    Each step becomes a dictionary with list of antecedents and consequent:    
    antecedents: step antecendents or left-hand side of implication
    consequent: step consequents or right-hand side of implication
    
    - ignore_broken_step: if True, then steps poorly formated will be
        skipped and the function will also return the list "broken_steps".
    - enforce_multi_antecedent: proof steps with less than two antecedents will be 
        marked as broken
    - enforce_dag_structure: ensure leaves are "sentX" nodes, conclusions nodes are
        either "intX" or "hypothesis", and every step represents a different node
        with increasing order (i.e. int1, int2, ..., hypothesis)
    - enforce_tree_structure: enforces structure not only is a DAG but also a 
        tree (every node has at most one parent), if set to True implies that it will 
        also assume "enforce_dag_structure" as True (all trees are DAGs)
    - enforce_nodes_in_contex: ensures that all nodes used are part of context (i.e. if a node
        not in the context or intermediate steps are referenced, then step is broken). 
        This will have no effect if context is None
    '''
    parsed_proof = {
        'steps': []
    }
    
    split_idxs = [m.span()[0]+1 for m in re.finditer(';[ ]+(int|sent)[0-9]+', proof)]
    split_idxs = [0] + split_idxs
    steps = [proof[i:j][:-1] for i,j in zip(split_idxs, split_idxs[1:]+[None])]
    
    steps = [s.strip() for s in steps]       
    
    # get context sentences to use in parsed proof
    symbol_to_text = {}
    if context:
        symbol_to_text.update(
            {k: v['text'] for k, v in parse_reasoning_context(context).items()})
    if hypothesis:
        symbol_to_text['hypothesis'] = hypothesis

    broken_steps = []
    for step in steps:
        try:
            # parse step
            parsed_step, symbol_to_text = parse_reasoning_step(
                step, symbol_to_text = symbol_to_text,
                enforce_multi_antecedent = enforce_multi_antecedent)
            
            # verify step follows constraints
            if (enforce_dag_structure and 
                not proof_step_follows_dag_structure(parsed_step,
                                                     parsed_proof['steps'])):
                raise ValueError(
                    (f'Step does not follow a DAG structure!\nstep = {step}'
                     f'\nparsed_step = {parsed_step}\nproof = {proof}'))
            if (enforce_tree_structure and 
                not proof_step_follows_tree_structure(parsed_step, 
                                                      parsed_proof['steps'])):
                raise ValueError(
                    (f'Step does not follow tree structure!\nstep = {step}'
                     f'\nparsed_step = {parsed_step}\nproof = {proof}'))
            if (context is not None and enforce_nodes_in_contex and 
                not proof_step_nodes_in_context(parsed_step, symbol_to_text.keys())):
                raise ValueError(
                    (f'Step has symbols not in context!\nstep = {step}'
                     f'\nparsed_step = {parsed_step}\nproof = {proof}'
                     f'\ncontext= {context}\nsymbols = {symbol_to_text.keys()}'))
            if (context is not None and enforce_finished_uses_all_contex and                
                not finished_proof_uses_all_context(parsed_step, parsed_proof['steps'], 
                                                    symbol_to_text.keys())):
                raise ValueError(
                    (f'Finished proof did not use all leaves in context!'
                     f'\nstep = {step}\nparsed_step = {parsed_step}\nproof = {proof}'
                     f'\ncontext= {context}\nsymbols = {symbol_to_text.keys()}'))
            
            # add step to parsed proof
            parsed_proof['steps'].append(parsed_step)
        except Exception as e:
            if ignore_broken_step:
                broken_steps.append((step, e))
            else:
                raise e
    
    possible_hypothesis = [
        step['consequent'] for step in parsed_proof['steps']
        if 'hypothesis' in step['consequent']['symbol']]
    if len(possible_hypothesis) == 0:
        # proof didn't reach hypothesis, try using hypothesis given as input
        if hypothesis is not None:
            parsed_proof['hypothesis'] = {
                'symbol': 'hypothesis',
                'text': hypothesis
            }
    elif len(possible_hypothesis) == 1:
        parsed_proof['hypothesis'] = possible_hypothesis[0]
    
    parsed_proof['intermediates'] = [
        step['consequent'] for step in parsed_proof['steps']
        if step['consequent']['symbol'] != 'hypothesis']
    parsed_proof['leaves'] = [
        ant for step in parsed_proof['steps'] 
        for ant in step['antecedents']
        if ant not in parsed_proof['intermediates']]
    parsed_proof['symbol_to_text'] = symbol_to_text
    
    if ignore_broken_step:
        return parsed_proof, broken_steps
    
    return parsed_proof