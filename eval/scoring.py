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
import itertools
import networkx as nx
from collections import defaultdict, Counter

from eval.parsing import (
    is_intermediate_node_symbol,
    parse_reasoning_proof,
)

####################################################################
# Proof comparison (see if a proof is equal or subproof of another)

def parsed_proof_to_dict_representation(parsed_proof):
    '''
    transform proof in a DAG representation.
    
    it will be a list of subproofs (all of which a dictionary)
    '''
    parent_to_children = defaultdict(list)
    not_root_set = set()
    for step in parsed_proof['steps']:
        con_symb = step['consequent']['symbol']
        for ant in step['antecedents']:
            ant_symb = ant['symbol']
            parent_to_children[con_symb].append(ant_symb)
            not_root_set.add(ant_symb)
    
    all_root_nodes = set(parent_to_children.keys()) - not_root_set
    subproofs_lst = []
    
    def get_all_leaves(node, parent_to_children):
        if node in parent_to_children:
            leaves = [get_all_leaves(c, parent_to_children)
                      for c in parent_to_children[node]]
            return sorted([e for l in leaves for e in l])
        return [node]
    
    def get_subproof(node, parent_to_children):
        if node in parent_to_children:
            subproofs = [get_subproof(c, parent_to_children)
                          for c in parent_to_children[node]]
            sort_k = lambda x: get_all_leaves(list(x.keys())[0], 
                                              parent_to_children) \
                if type(x) == dict else [x]
            subproofs = sorted(subproofs, key=sort_k)            
            return {node: subproofs}
        return node
    
    for root_node in all_root_nodes:
        try:
            subproofs_lst.append(get_subproof(root_node, parent_to_children))
        except Exception as e:
            # TODO: should figure out why this is happening.
            print('\n\nERROR ON parsed_proof_to_dict_representation!!!!')
            print('ERROR = ', str(e))
            print(f'parsed_proof = {parsed_proof}')
            print(f'parent_to_children = {parent_to_children}')
            print(f'all_root_nodes = {all_root_nodes}')
            print('\n\n')
            return None
    
    # clear memory
    get_all_leaves = None
    get_subproof = None
    
    return subproofs_lst

def parsed_proof_is_subproof(pp_1, pp_2):
    '''
    Return True if and only if pp_1 is a subtree of pp_2.
    
    In this case the sentX nodes and hypothesis nodes should match, and the intX
    nodes are allowed to have any matching.
    
    Also returns a list of corresponding subproof from pp_3 that matches 
    the subproofs from pp_1
    '''
    
    subproofs_lst_1 = parsed_proof_to_dict_representation(pp_1)
    subproofs_lst_2 = parsed_proof_to_dict_representation(pp_2)
    
    # sort subproofs by size 
    subproofs_lst_1 = sorted(subproofs_lst_1, key = lambda x: len(str(x)))
    subproofs_lst_2 = sorted(subproofs_lst_2, key = lambda x: len(str(x)))

    if subproofs_lst_1 is None or subproofs_lst_2 is None:
        # problem transforming to dict representation
        return False, []
    
    def ignore_subproof_int_num(subproofs_lst):
        # convert a dict representation of proof by substituting
        # all 'intX' to 'int' (without the number)
        if type(subproofs_lst) == dict:
            node_symb = list(subproofs_lst.keys())[0]
            if is_intermediate_node_symbol(node_symb):
                node_symb = 'int'
            return {node_symb: [ignore_subproof_int_num(s) 
                                for s in subproofs_lst.values()]}
        if type(subproofs_lst) == list:
            return [ignore_subproof_int_num(p) for p in subproofs_lst]
        return subproofs_lst
    
    def matches_subproof(p_sub, p_super):
        # returns None if doesn't match, or a subproof of p_super 
        # if it matches p_sub (assumes all intermediate nodes are equal)
        if ignore_subproof_int_num(p_sub) == ignore_subproof_int_num(p_super):
            return p_super
        if type(p_super) == dict:
            p_super = list(p_super.values())
        if type(p_super) == list:
            # sort subproofs by size (hack to handle cases when
            # p_sub can match multiple sub proofs in p_super)
            p_super = sorted(p_super, key = lambda x: len(str(x)))
            matches = [matches_subproof(p_sub, p) for p in p_super]
            matches = list(filter(None, matches))
            return matches[0] if len(matches) > 0 else None
        return None

    def get_all_nodes(p_dict):
        # returns a list of intermediate nodes from proof dict representation
        if type(p_dict) == list:
            return [y for x in list(map(get_all_nodes, p_dict)) for y in x]
        if type(p_dict) == dict:
            child_lst = list(p_dict.values())[0]
            root_node = list(p_dict.keys())
            return root_node + [n for p in child_lst for n in get_all_nodes(p)]
        return [p_dict]
    
    match_lst = []
    available_nodes = Counter(get_all_nodes(subproofs_lst_2))
    
    for subp_1 in subproofs_lst_1:
        found = False
        
        for subp_2 in subproofs_lst_2:
            match_subp_2 = matches_subproof(subp_1, subp_2)
            matched_nodes = Counter(get_all_nodes(match_subp_2))
            # ensure subproof one to one mapping, test if match
            # uses nodes that are not available
            is_match_used = any(
                [matched_nodes[k] > v 
                for k, v in available_nodes.items()]
            )
            if match_subp_2 and not is_match_used:
                match_lst.append((subp_1, match_subp_2))
                available_nodes.subtract(matched_nodes)
                found = True
                break
            
        if not found:
            # every "sub proof" in pp_1 has to be part of a subproof in pp_2
            return False, []
    
    # clear memory
    ignore_subproof_int_num = None
    matches_subproof = None

    return True, match_lst

def proof_is_subproof(proof1, proof2, similarity_fn = None):
    '''
    Return True if and only if proof1 is a subtree (subproof) of proof2.
    
    In this case the sentX nodes and hypothesis nodes should match, and the intX
    nodes are allowed to have any matching.
    
    if similarity_fn is provided, will also test text similarity of every matching 
    subproof root and intermediate nodes.
    '''
    pp1, broken1 = parse_reasoning_proof(proof1, ignore_broken_step = True)
    pp2, broken2 = parse_reasoning_proof(proof2, ignore_broken_step = True)

    if len(broken1) > 0:
        return False
    # assumes target proof is correctly formated
    assert len(broken2) == 0
    
    symb_to_txt1 = pp1['symbol_to_text']
    symb_to_txt2 = pp2['symbol_to_text']
    is_match, match_lst = parsed_proof_is_subproof(pp1, pp2)
    
    def is_text_similar(match, symb_to_txt1, symb_to_txt2, similarity_fn):
        # given a match object (a tuple of proof dict representation) will
        # test if the text of each subproof is similar according to the
        # given similarity_fn
        mp1, mp2 = match
        if type(mp1) != dict or type(mp2) != dict:
            # In this case no intermediate node to compare
            return True
        m_symb_1 = list(mp1.keys())[0]
        m_symb_2 = list(mp2.keys())[0]
        # must be intermediate node
        if (m_symb_1 != 'hypothesis' and not 
            similarity_fn(symb_to_txt1[m_symb_1], symb_to_txt2[m_symb_2])):
            # print('proof_is_subproof NOT TEXT SIMILAR:', 
            #       symb_to_txt1[m_symb_1], ' // ', symb_to_txt2[m_symb_2])
            return False
        sub_matches = [x for x in zip(list(mp1.values())[0], 
                                      list(mp2.values())[0])]
        for sub_match in sub_matches:
            if not is_text_similar(sub_match, symb_to_txt1, 
                                   symb_to_txt2, similarity_fn): 
                return False
        return True
    
    # main code for comparison
    if is_match and similarity_fn is not None:
        for match in match_lst:
            if not is_text_similar(match, symb_to_txt1, 
                                   symb_to_txt2, similarity_fn):
                return False
    return is_match

def proof_is_equivalent(proof1, proof2, similarity_fn = None):
    '''
    Tests if two proofs are equivalent (i.e. they are subproofs of each other)
    
    if similarity_fn is provided, will also test text similarity of every matching 
    subproof root and intermediate nodes.
    '''
    sub_1_2 = proof_is_subproof(proof1, proof2, similarity_fn = similarity_fn) 
    sub_2_1 = proof_is_subproof(proof2, proof1, similarity_fn = similarity_fn)
    return sub_1_2 and sub_2_1


####################################################################
# DAG proof comparison

def from_parsed_proof_to_nxgraph(pp):
    '''
    Convert parsed proof to nx graph
    '''
    nx_g = nx.DiGraph()
    all_nodes = []
    all_edges = []
    symb_to_text = pp['symbol_to_text']
    for step in pp['steps']:
        to_node = step['consequent']['symbol']
        to_node_text = ''
        if to_node in symb_to_text:
            to_node_text = symb_to_text[to_node]
        all_nodes.append((to_node, to_node_text))
        for antecedent in step['antecedents']:
            from_node = antecedent['symbol']
            from_node_text = ''
            if from_node in symb_to_text:
                from_node_text = symb_to_text[from_node]
            all_nodes.append((from_node, from_node_text))
            all_edges.append((from_node, to_node))
    
    all_nodes = list(set(all_nodes))
    for n_idx, (symb, text) in enumerate(all_nodes):
        nx_g.add_node(symb, symbol=symb, text=text)
    nx_g.add_edges_from(all_edges)
    return nx_g

def proof_edit_distance(proof1, proof2, similarity_fn = None, return_similarity = True):
    '''
    Compute the graph edit distance from proof1 to proof2
    
    if 'return_similarity' is true, the distance will be divided by the maximum number of possible 
    edits between proof1 and proof2.
    
    if similarity_fn is provided, will also test text similarity of every matching 
    graph node.
    
    NOTE: this can be very computationally expensive if the graphs are
    large (exponential time complexity)
    '''
    pp1, broken1 = parse_reasoning_proof(proof1, ignore_broken_step = True)
    pp2, broken2 = parse_reasoning_proof(proof2, ignore_broken_step = True)
    
    if len(broken1) > 0:
        return None
    # assumes target proof is correctly formated
    assert len(broken2) == 0
    
    def node_match(n1, n2):
        symb1 = n1['symbol']; symb2 = n2['symbol']
        if symb1.startswith('sent') or symb2.startswith('sent'):
            # premise node should always match according to symbol
            return symb1 == symb2 
        if similarity_fn:
            # conclusion nodes will match according to
            # symilarity_fn on text
            return similarity_fn(n1['text'], n2['text'])
        return True
    
    nx_g1 = from_parsed_proof_to_nxgraph(pp1)
    nx_g2 = from_parsed_proof_to_nxgraph(pp2)    
    
    optimized_edit_distances_gen = nx.optimize_graph_edit_distance(
            nx_g1, nx_g2, node_match=node_match)
    if nx_g1.number_of_nodes() + nx_g2.number_of_nodes() < 10:
        optimized_edit_distances = list(itertools.islice(
            optimized_edit_distances_gen, 3
        ))        
        edit_distance = min(optimized_edit_distances)
    else:
        # graph is too large
        edit_distance = next(optimized_edit_distances_gen)
    
    # edit_distance = sum(optimized_edit_distances) / len(optimized_edit_distances)    
    # edit_distance = nx.graph_edit_distance(nx_g1, nx_g2, node_match=node_match, timeout=5)   

    if return_similarity:
        g1_sz = nx_g1.number_of_nodes() + nx_g1.number_of_edges()
        g2_sz = nx_g2.number_of_nodes() + nx_g2.number_of_edges()
        # normalize graph
        graph_similarity = float(edit_distance) / float(max(g1_sz, g2_sz))
        if graph_similarity > 1.0:
            # print('EDIT DISTANCE ERROR! graph_similarity < 0')
            # print(f'g1_sz = {g1_sz}')
            # print(f'g2_sz = {g2_sz}')
            # print(f'edit_distance = {edit_distance}')
            # print(f'graph_similarity = {graph_similarity}')
            graph_similarity = min(graph_similarity, 1.0)
        assert graph_similarity <= 1.0
        # inverse distance
        graph_similarity = 1.0 - graph_similarity
        return graph_similarity
    return edit_distance