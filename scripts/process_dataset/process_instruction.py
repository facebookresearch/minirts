# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from collections import defaultdict

import os
import sys
script_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(script_root, 'behavior_clone'))
from inst_dict import InstructionDict, correct_instruction, clean_instruction


def calc_edit_distance(s, t):
    m, n = len(s) + 1, len(t) + 1
    f = {}
    for i in range(m):
        f[i, 0] = i
    for j in range(n):
        f[0, j] = j
    for i in range(1, m):
        for j in range(1, n):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            f[i, j] = min(f[i, j - 1] + 1, f[i - 1, j] + 1, f[i - 1, j - 1] + cost)
    return f[m - 1, n - 1]


def get_all_instructions(entrys):
    insts = []
    for entry in entrys:
        inst = entry['instruction']
        if len(insts) == 0 or insts[-1] != inst:
            insts.append(inst)

    return insts


def get_unique_instructions_and_count(entrys):
    insts = get_all_instructions(entrys)
    counts = defaultdict(int)
    for inst in insts:
        counts[inst] += 1
    return counts


def get_unique_word_and_count(instructions):
    counts = defaultdict(int)
    for inst in instructions:
        inst = clean_instruction(inst)
        words = inst.split()
        for word in words:
            word = word.strip()
            if len(word) == 0:
                continue
            counts[word] += 1

    return counts


def filter_by_freq(counts, min_freq, prefix):
    new_counts = {}
    for word, count in counts.items():
        if count >= min_freq:
            new_counts[word] = count

    print('>>>filter %s by freq: >=%d' % (prefix, min_freq))
    print('\tbefore filtering: ', len(counts))
    print('\tafter filtering:', len(new_counts))
    return new_counts


def filter_topk(counts, k):
    sorted_counts = list(counts.items())
    sorted_counts = sorted(sorted_counts, key=lambda x: -x[1])
    topk = sorted_counts[:k]
    rest = sorted_counts[k:]
    new_counts = dict(topk)
    return new_counts, topk, rest


def find_close_word(source, targets, edit_dist):
    min_dist = 100
    best_match = ''
    for target in targets:
        if target == source:
            continue
        d = calc_edit_distance(source, target)
        if d < min_dist:
            min_dist = d
            best_match = target

    if min_dist <= edit_dist:
        return best_match
    else:
        return None


#  merge those appear in top100
special_corrections = {
    'mines': 'mine',
    'peasants': 'peasant',
    'catapults': 'catapults',
    'dragons': 'dragon',
    'minerals': 'mineral',
    'spearmen': 'spearman',
    'towers': 'tower',
    'archers': 'archer',
}


def create_correction_dict(counts, topk, edit_dist, cut_off_freq):
    corrections = {}
    correct2wrongs = defaultdict(list)
    topk_counts, topk, rest = filter_topk(counts, topk)
    topk_words = [w[0] for w in topk]
    for word, freq in rest:
        if len(word) <= 2:
            continue

        if len(word) <= 4 or freq >= cut_off_freq:
            max_dist = 1
        else:
            max_dist = edit_dist

        best_match = find_close_word(word, topk_words, max_dist)
        if best_match is not None:
            # print('correction: %s -> %s' % (word, best_match))
            corrections[word] = best_match
            correct2wrongs[best_match].append(word)

    return corrections, correct2wrongs


def correct_entrys(entrys, corrections=None):
    """entry point for all other scripts"""
    if corrections is None:
        insts = get_all_instructions(entrys)
        counts = get_unique_word_and_count(insts)
        corrections, correct2wrongs = create_correction_dict(counts, 100, 2, 5)
        corrections.update(special_corrections)

    for entry in entrys:
        inst = entry['instruction']
        entry['instruction'] = correct_instruction(inst, corrections)

    return entrys, corrections


def create_dictionary(entrys, corrections, word_min_freq, inst_min_freq):
    inst_counts = get_unique_instructions_and_count(entrys)
    word_counts = get_unique_word_and_count(inst_counts.keys())

    filtered_inst_counts = filter_by_freq(inst_counts, inst_min_freq, 'instruction')
    filtered_word_counts = filter_by_freq(word_counts, word_min_freq, 'word')

    inst_dict = InstructionDict(filtered_word_counts, filtered_inst_counts, corrections)
    return inst_dict


if __name__ == '__main__':
    import pickle
    import argparse
    import json

    parser = argparse.ArgumentParser(description='process instruction')
    parser.add_argument('--states-root', type=str)
    parser.add_argument('--input-dataset1',
                        type=str,
                        default='data/new_train.json_min10_processed')
    parser.add_argument('--input-dataset2',
                        type=str,
                        default='data/new_valid.json_min10_processed')
    parser.add_argument('--instructions', type=str, default='data/new_insts.pt')
    args = parser.parse_args()

    # insts = pickle.load(open(args.instructions, 'rb'))
    # counts = get_unique_word_and_count(insts)
    # corrections, correct2wrongs = create_correction_dict(counts, 100, 2, 5)
    # corrections.update(special_corrections)

    dataset1 = json.load(open(args.input_dataset1))
    dataset2 = json.load(open(args.input_dataset2))
    entrys = dataset1 + dataset2
