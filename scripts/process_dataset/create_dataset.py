# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import sys
import random
import json
import argparse
import pprint
import pickle
from copy import deepcopy

from tqdm import tqdm
import numpy as np

from common_utils import Logger
from utils import get_all_files
from analyze_dataset import filter_bad_replays
import common_utils.global_consts as gc
from process_instruction import *
import inst_dict


def label_all_cont(game):
    """
    for a frame in game, if all units have no new target, this frame
    is mark as glob_cont = True
    """
    num_all_cont = 0
    for frame in game:
        if frame is None:
            continue

        all_cont = True
        for unit in frame['my_units']:
            if unit['target_cmd']['cmd_type'] != gc.CmdTypes.CONT.value:
                all_cont = False

        frame['glob_cont'] = int(all_cont)
        num_all_cont += all_cont

    return game


def filter_beginning(game):
    # print('filtering all-cont at beginning')
    # filtered_count = []
    new_game = []
    i = 0
    while i < len(game):
        instruction = game[i]['instruction']
        j = i

        # filter beginning
        must_include = False
        while j < len(game) and game[j]['instruction'] == instruction:
            if must_include:
                new_game.append(game[j])
                j += 1
                continue

            all_cont = True
            for unit in game[j]['my_units']:
                target_cmd_type = unit['target_cmd']['cmd_type']
                if target_cmd_type != gc.CmdTypes.CONT.value:
                    all_cont = False

            if not all_cont:
                must_include = True
                new_game.append(game[j])
                # filtered_count.append(j-i)
                # if j - i > 10:
                #     print(j, game[j]['instruction'])
            j += 1
        i = j

    return new_game


def add_prev_instruction(game):
    """
    prev_instruction is used by coach as 'current_instruction'
    """
    if len(game) == 0:
        assert False, 'empty game'

    game[0]['prev_instruction'] = ''
    for i in range(1, len(game)):
        game[i]['prev_instruction'] = game[i-1]['instruction']

    return game


def add_prev_cmd(game):
    """
    add prev_cmd field. Note that this field is different for
    instructor and executor (off by 1 when new instruction is issued)
    """
    i = 0
    unit2prev_cmds = defaultdict(list)

    while i < len(game):
        inst = game[i]['instruction']
        j = i
        while j < len(game) and game[j]['instruction'] == inst:
            for unit in game[j]['my_units']:
                unit_id = unit['unit_id']
                if len(unit2prev_cmds[unit_id]) == 0:
                    # new unit
                    unit2prev_cmds[unit_id].append(unit['current_cmd']['cmd_type'])

                unit['prev_cmd'] = deepcopy(unit2prev_cmds[unit_id])

                if 'target_cmd' not in unit:
                    assert False, 'unit should be pre-processed with add_target'

                target_cmd = unit['target_cmd']
                cmd_type = target_cmd['cmd_type']
                unit2prev_cmds[unit_id].append(cmd_type)
            j += 1
        i = j
    return game


def split_train_val(files, val_ratio, seed):
    train_files = []
    val_files = []

    random.seed(seed)
    for f in files:
        if random.random() < val_ratio:
            val_files.append(f)
        else:
            train_files.append(f)

    return train_files, val_files


def create_dataset(files):
    data = []
    for f in tqdm(files):
        game = json.load(open(f, 'r'))
        game = label_all_cont(game)
        game = filter_beginning(game)
        game = add_prev_instruction(game)
        game = add_prev_cmd(game)

        data.extend(game)
    return data


def add_base_frame(dataset):
    """
    add base_frame field. Note that this field is different for
    instructor and executor (off by 1 when new instruction is issued)
    """
    coach_base_frame = 0
    last_replay = dataset[0]['unique_id'].rsplit('-', 1)[0]

    i = 0
    while i < len(dataset):
        inst = dataset[i]['instruction']
        replay = dataset[i]['unique_id'].rsplit('-', 1)[0]
        if replay != last_replay:
            last_replay = replay
            coach_base_frame = i

        j = i
        replay_ = dataset[j]['unique_id'].rsplit('-', 1)[0]
        while (j < len(dataset)
               and dataset[j]['instruction'] == inst
               and replay_ == replay):
            dataset[j]['coach_base_frame_idx'] = coach_base_frame
            dataset[j]['executor_base_frame_idx'] = i

            if j == i:
                coach_base_frame = j

            j += 1
        i = j
    return dataset


def create_dictionary_and_correct_instruction(trainset, valset):
    trainset, corrections = correct_entrys(trainset, None)
    if valset is not None:
        valset, _ = correct_entrys(valset, corrections)
    inst_dict = create_dictionary(trainset, corrections, 5, 0)
    return inst_dict, trainset, valset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=99)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--min_num_target', type=int, default=0)
    parser.add_argument('--min_num_instruction', type=int, default=0)
    parser.add_argument('--raw_json_root', type=str, required=True,
                        help='used to decide which to filter')
    parser.add_argument('--processed_json_root', type=str, required=True,
                        help='used to create the dataset')
    parser.add_argument('--output', type=str, required=True)

    args = parser.parse_args()
    logger_path = os.path.join(args.output, 'config')
    sys.stdout = Logger(logger_path)

    print('configs:')
    pprint.pprint(vars(args))

    raw_json = get_all_files(args.raw_json_root, '.json')
    processed_json = get_all_files(args.processed_json_root, '.json')
    if len(raw_json) != len(processed_json):
        print(len(raw_json), len(processed_json))
        assert False

    print('filtering with min_target=%d, min_inst=%d'
          % (args.min_num_target, args.min_num_instruction))
    good_json, bad_json = filter_bad_replays(
        raw_json, args.min_num_instruction, args.min_num_target)
    print('removed %d json files' % len(bad_json))

    processed_json = [f.replace(args.raw_json_root, args.processed_json_root)
                      for f in good_json]
    train_files, val_files = split_train_val(
        processed_json, args.val_ratio, args.seed)

    trainset = create_dataset(train_files)
    add_base_frame(trainset)
    print('len(trainset) =', len(trainset))
    if len(val_files):
        valset = create_dataset(val_files)
        add_base_frame(valset)
        print('len(valset) =', len(valset))
    else:
        valset = None

    inst_dict, trainset, valset = \
        create_dictionary_and_correct_instruction(trainset, valset)
    pickle.dump(inst_dict, open(os.path.join(args.output, 'dict.pt'), 'wb'))

    print('writing dev to file')
    devset = trainset[:2000]
    with open(os.path.join(args.output, 'dev.json'), 'w') as f:
        json.dump(devset, f)

    if len(val_files):
        print('writing val to file')
        with open(os.path.join(args.output, 'val.json'), 'w') as f:
            json.dump(valset, f)
        with open(os.path.join(args.output, 'val_files.json'), 'w') as f:
            json.dump(val_files, f, indent=4)

    print('writing train to file')
    with open(os.path.join(args.output, 'train.json'), 'w') as f:
        json.dump(trainset, f)
    with open(os.path.join(args.output, 'train_files.json'), 'w') as f:
        json.dump(train_files, f, indent=4)
