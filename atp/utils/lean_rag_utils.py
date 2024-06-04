import os
from lean_dojo import *
from utils.lean_math_utils import *
import lean_dojo
from random import randint
import time

import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

class TripletDataset(Dataset):
    def __init__(self, triplets, tokenizer):
        self.triplets = triplets
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index):
        q, s, n = self.triplets[index]
        q_encoded = self.tokenizer(q, padding='max_length', max_length=256, truncation=True, return_tensors='pt')
        s_encoded = self.tokenizer(s, padding='max_length', max_length=64, truncation=True, return_tensors='pt')
        n_encoded = self.tokenizer(n, padding='max_length', max_length=64, truncation=True, return_tensors='pt')
        return q_encoded, s_encoded, n_encoded

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, q_embed, s_embed, n_embed):
        dist_pos = torch.norm(q_embed - s_embed, p=2, dim=1)
        dist_neg = torch.norm(q_embed - n_embed, p=2, dim=1)
        loss = torch.mean(torch.relu(dist_pos - dist_neg + self.margin))
        return loss
    
# Example query string

def get_similar_tacs(query_string, model_state, tokenizer, faiss_index, all_tacs, num_returned=100, verbose=False):
    # Convert query string to BERT embedding
    device = model_state.device
    with torch.no_grad():
        encoded_input = tokenizer(query_string, padding=True, truncation=True, return_tensors='pt').to(device)
        outputs = model_state(**encoded_input)
        query_embedding = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()

    # Perform nearest neighbor search
    distances, indices = faiss_index.search(query_embedding, num_returned)

    if verbose:
        # Print the nearest neighbors and their distances
        print(f"Nearest neighbors for: {query_string}")
    output = []
    for i in range(num_returned):
        idx = indices[0][i]
        distance = distances[0][i]
        similarity = 1.0 / (1 + distance*distance/1000)
        tac = all_tacs[idx]
        output.append((all_tacs[idx],similarity))
        if verbose:
            print(f"{i+1}. {all_tacs[idx]} (similarity: {similarity})")
    return sorted(output, key=lambda item: item[1], reverse=True)




MAX_STEPS = 100000
MAX_TACTIC_FROM_TEMPLATE = 50
PENALTY_SEEN_TARGET_MULTIPLIER = 3
MAX_NUM_DOJO_ATTEMPT = 3

#theorem = Theorem(repo, "MIL/C02_Basics/solutions/Solutions_S05_Proving_Facts_about_Algebraic_Structures.lean",
#                  "Solutions_S05_Proving_Facts_about_Algebraic_Structures_ex1")
# theorem = Theorem(repo, "MIL/C02_Basics/solutions/Solutions_S01_Calculating.lean",
#                   "Solutions_S01_Calculating_ex2")
#theorem = Theorem(repo, "MIL/C02_Basics/solutions/Solutions_S04_More_on_Order_and_Divisibility.lean",
#                  "C02S04.Solutions_S04_More_on_Order_and_Divisibility_ex7")


# The complexity of a state if calculated from lengths of its targets
# seen_target_freq is a dict{string, int} of strings containing lines of targets that have been seen
# If all targets in the current state has been seen, add to complexity PENALTY_SEEN_TARGET_MULTIPLIER*min_freq
# At the same time, seen_target_freq is updated by the targets in this state
# if base_complexity is not none, complexity is base_complexity+1
def explore_state_complexity(state, base_complexity=None, seen_target_freq=None):
    if '⊢ False' in state.pp:
        return 1000000
    if base_complexity is not None:
        return base_complexity + 1
    complexity = 0
    lines = state.pp.split('\n')
    targets = []
    min_freq = 1000000
    for line in lines:
        if line.startswith("⊢"):
            target = line[1:].strip()
            targets.append(target)
            if seen_target_freq is not None:
                if target in seen_target_freq:
                    min_freq = min(min_freq, seen_target_freq[target])
                    seen_target_freq[target] = seen_target_freq[target] + 1
                else:
                    min_freq = 0
                    seen_target_freq[target] = 1
    lengths = [len(x) for x in targets]
    complexity = max(lengths)/2 + sum(lengths)/2
    if seen_target_freq is not None:
        complexity += PENALTY_SEEN_TARGET_MULTIPLIER*min_freq
    return complexity

# Print the trace from state_0 to state
def get_tactic_trace(curr_state, state_dict):
    _, parent_states, tactics = state_dict[curr_state]
    if len(parent_states) == 0:
        return []
    else:
        return get_tactic_trace(parent_states[0], state_dict) + [(tactics[0], curr_state)]

def explore_states(theorem,
                   model_state,
                   tokenizer,
                   rag_index,
                   all_tacs, 
                   theorem_code,
                   traced_tactics=None,
                   max_steps=MAX_STEPS, 
                   max_time=1200, 
                   exit_on_finish=True, 
                   verbose=False):
    start_time = time.time()
    # For some theorems, it might take a few minutes.
    dojo, state_0 = Dojo(theorem).__enter__()

    #if verbose: print(state_0.pp)
    unused_vars = []
    if theorem_code is not None:
        tokens = tokenize_lean_tactic(theorem_code)
        tokens = [x for x in tokens if x.strip() != '']
        type_of_item, def_of_item = classify_lean_elements(state_0.pp)
        unused_vars = [x for x in type_of_item.keys() if x not in tokens]
        #print(unused_vars)

    state_queue = PriorityQueue() # PQ of states, priority being complexity of the state
    state_dict = {} # Keyed by state.pp (or state if it is ProofFinished). Value is (state, its parent states, and tactics from parent state to this state)

    curr_state = state_0
    base_complexity = 0
    state_queue.push(curr_state, explore_state_complexity(curr_state, base_complexity=base_complexity) + randint(0, 4))
    base_complexity += 1
    state_dict[curr_state.pp] = (curr_state, list(), list())
    
    # Follow traced_tactics and populate state_dict with the authoratative proof
    if traced_tactics is not None:
        for traced_tactic in traced_tactics:
            tactic = traced_tactic.tactic
            test_state = dojo.run_tac(curr_state, tactic)
            if type(test_state) == lean_dojo.interaction.dojo.LeanError:
                break
            elif type(test_state) == lean_dojo.interaction.dojo.ProofFinished:
                state_dict[test_state] = (test_state, [curr_state], [tactic])
                if verbose: print("Sucessfully followed proof")
                break
            else:
                state_dict[test_state.pp] = (test_state, [curr_state], [tactic])
                curr_state = test_state
    
    n_steps = 0
    theorem_proven = False
    while state_queue.size() > 0:
        curr_state = state_queue.pop()
        if not hasattr(curr_state, 'pp'):
            continue
        #print("POP STATE: ", base_complexity, curr_state.pp, "\nEND STATE")
        type_of_item, def_of_item = classify_lean_elements(curr_state.pp)
        for var in unused_vars:
            if var in type_of_item:
                del type_of_item[var]
            if var in def_of_item:
                del def_of_item[var]
        if verbose:
            if n_steps == 0:
                print("INIT_STATE:", curr_state.pp)
                print(type_of_item)
                #print(def_of_item)
        #
        #TODO: Add {'nvar0': 'nvar0', 'nvar1': 'nvar1', ...} to type_of_item
        #
        tactic_set = set()
        query = theorem_code + ' # ' + curr_state.pp
        suggestions = get_similar_tacs(query, model_state, tokenizer, rag_index, all_tacs, num_returned=200)
        tac_templates = [x[0] for x in suggestions]
        for tac_template in tac_templates:
            try:
                tactics = generate_tactics_from_template(tac_template, type_of_item)
                if len(tactics) > MAX_TACTIC_FROM_TEMPLATE:
                    tactics = random.sample(tactics, MAX_TACTIC_FROM_TEMPLATE)
            except:
                #print("Cannot generate tactics using", tac_template, ":", type_of_item)
                continue
            for tactic in tactics:
                tactic_set.add(tactic)
        #
        proof_finished = False
        tactic_trace = None
        #print(len(tactic_set), "tactics")
        #print(tactic_set)
        for tactic in tactic_set:
            num_attempt = 0
            while num_attempt < MAX_NUM_DOJO_ATTEMPT:
                try:
                    test_state = dojo.run_tac(curr_state, tactic)
                    break
                except:
                    num_attempt += 1
            if num_attempt == MAX_NUM_DOJO_ATTEMPT:
                continue
            #print("TRY TAC:", tactic)
            n_steps += 1
            if n_steps % 10000 == 0:
                print(n_steps, "steps executed")
            if n_steps > max_steps:
                break
            if type(test_state) == lean_dojo.interaction.dojo.LeanError:
                continue
            elif type(test_state) == lean_dojo.interaction.dojo.ProofFinished:
                proof_finished = True
                if verbose and not theorem_proven:
                    print("ProofFinished")
#                     tactic_trace = get_tactic_trace(curr_state, state_dict) + [(tactic, test_state)]
#                     for i in range(len(tactic_trace)):
#                         tactic, state = tactic_trace[i]
#                         print("STEP", i)
#                         if tactic != None:
#                             print("Tactic:", tactic)
#                         if hasattr(state, 'pp'):
#                             print(state.pp)
                if test_state in state_dict:
                    _, parent_states, tactics = state_dict[test_state]
                    state_dict[test_state] = (test_state, parent_states + [curr_state], tactics + [tactic])
                else:
                    complexity = 100000000
                    state_queue.push(test_state, complexity)
                    state_dict[test_state] = (test_state, [curr_state], [tactic])
                if exit_on_finish: break
            else:
                #print("TAC:", tactic, "STATE:", test_state.pp)
                if test_state.pp in state_dict:
                    _, parent_states, tactics = state_dict[test_state.pp]
                    state_dict[test_state.pp] = (test_state, parent_states + [curr_state], tactics + [tactic])
                else:
                    complexity = explore_state_complexity(test_state, base_complexity=base_complexity)
                    base_complexity += 1
                    state_queue.push(test_state, complexity  + randint(0, 4))
                    state_dict[test_state.pp] = (test_state, [curr_state], [tactic])
        #
        if proof_finished:
            theorem_proven = True
            if exit_on_finish: break
        if n_steps > max_steps:
            break
        cur_time = time.time()
        if (cur_time - start_time) > max_time:
            print("Max run-time exceeded")
            break
    #
    dojo.__exit__(None, None, None)
    return state_dict, theorem_proven, tactic_trace

# Generate state pairs and their distances. 
# Probably useful if we hope to generate intermediate states (e.g., "have").

import random

def yield_state_pairs(curr_state, leaf_state, state_dict, seen_states_pp, tactic_list, max_distance=8, max_parents=10):
    #     print("curr_state:", curr_state.pp)
    #     print("leaf_state:", leaf_state.pp)
    if type(curr_state) == lean_dojo.interaction.dojo.ProofFinished:
        _, parent_states, tactics = state_dict[curr_state]
    else:
        _, parent_states, tactics = state_dict[curr_state.pp]
    if len(seen_states_pp) > 1:
        yield (curr_state, leaf_state, len(seen_states_pp) - 1, tactic_list)
    if len(seen_states_pp) <= max_distance:
        num_parents_checked = 0
        for i in range(len(parent_states)):
            parent_state = parent_states[i]
            if parent_state.pp not in seen_states_pp:
                yield from yield_state_pairs(parent_state, 
                                             leaf_state, 
                                             state_dict, 
                                             seen_states_pp + [parent_state.pp],
                                             [tactics[i]] + tactic_list
                                            )
                num_parents_checked += 1
                if num_parents_checked >= max_parents:
                    break
    return


def get_state_provability_data(theorem,
                               model_state,
                               tokenizer,
                               rag_index, 
                               all_tacs,
                               theorem_code=None, 
                               traced_tactics=None,
                               max_steps=100000, 
                               negative_ratio=1.0, 
                               verbose=False):
    state_dict, theorem_proven, tactic_trace = \
    explore_states(theorem,
                   model_state,
                   tokenizer,
                   rag_index, 
                   all_tacs,
                   theorem_code=theorem_code, 
                   traced_tactics=traced_tactics,
                   max_steps=max_steps,
                   exit_on_finish=False,
                   verbose=verbose)
    #
    all_states = list(state_dict.values())
    print(len(all_states), "states in total")
    proven_states = [x[0] for x in all_states if type(x[0]) == lean_dojo.interaction.dojo.ProofFinished]
    print(len(proven_states), "proven states")
    state_pairs = []
    seen_state_pps = set()
    #
    for proven_state in proven_states:
        print(proven_state)
        pairs = list(yield_state_pairs(proven_state, proven_state, state_dict, [str(proven_state)], []))
        ancestor_state_dict = {}
        for ancestor, leaf, distance, tactic in pairs:
            if ancestor.pp not in ancestor_state_dict:
                ancestor_state_dict[ancestor] = (distance, tactic)
            else:
                if distance < ancestor_state_dist[ancestor]:
                    ancestor_state_dict[ancestor] = (distance, tactic)
        #print(ancestor_state_dist)
        print(len(ancestor_state_dict), "ancestors found")
        for ancestor, (distance, tactic) in ancestor_state_dict.items():
            state_pairs.append((ancestor, proven_state, distance, tactic))
            seen_state_pps.add(ancestor.pp)
    #
    num_positive = len(state_pairs)
    num_negative = 0
    for tup in all_states:
        state = tup[0]
        if hasattr(state, 'pp') and state.pp not in seen_state_pps:
            state_pairs.append((state, random.choice(proven_states), -1, None))
            num_negative += 1
            if num_negative >= num_positive * negative_ratio:
                break
    return state_pairs, state_dict


def get_filename(filepath):
    # Extract the base name (e.g., 'example.txt' from '/path/to/example.txt')
    base_name = os.path.basename(filepath)
    # Split the base name and the extension and return just the base name
    file_name_without_extension = os.path.splitext(base_name)[0]
    return file_name_without_extension


def compute_provability_training_data(file_path,
                                      output_path,
                                      traced_repo,
                                      model_state,
                                      tokenizer,
                                      index,
                                      tacs
                                      ):
    file_name = get_filename(file_path)
    fout = open(output_path + file_name + '.pkl', 'wb')
    traced_file = traced_repo.get_traced_file(file_path)
    premises = traced_file.get_premise_definitions()
    #
    results = []
    for premise in premises:
        if premise['code'].startswith('theorem '):
            print('THEOREM:', premise['full_name'])
            print(premise['code'])
            thm = traced_file.get_traced_theorem(premise['full_name'])
            theorem = thm.theorem
            traced_tactics = thm.get_traced_tactics()
            print("theorem loaded")
            state_pairs, _ = \
                get_state_provability_data(theorem,
                                           model_state,
                                           tokenizer,
                                           index,
                                           tacs,
                                           theorem_code=premise['code'],
                                           traced_tactics=traced_tactics,
                                           max_steps=50000,
                                           verbose=True)
            print(len(state_pairs), "state pairs found")
            #
            for state_pair in state_pairs:
                result = (file_path, premise, theorem, state_pair)
                results.append(result)
                pickle.dump(result, fout)
    #
    fout.close()
    return len(results)
    
 