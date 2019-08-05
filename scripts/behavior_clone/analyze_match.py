# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
from collections import defaultdict
import os
import argparse
import pprint
import numpy as np


def average_across_seed(logs):
    new_logs = {} #defaultdict(list)
    for k, v in logs.items():
        s = k.rsplit('_', 1)
        if len(s) == 2:
            name, seed = s
        elif len(s) == 1:
            name = 'default'
            seed = s[0]
        if not seed.startswith('s'):
            print('no multiple seeds, omit averaging')

        if name not in new_logs:
            new_logs[name] = [0, 0, 0]

        if v is None:
            continue

        new_logs[name][0] += v[0]
        new_logs[name][1] += v[1]
        new_logs[name][2] += v[2]

    for k in new_logs:
        vals = new_logs[k]
        total_game = np.sum(vals)
        new_logs[k][0] /= total_game
        new_logs[k][1] /= total_game
        new_logs[k][2] /= total_game

        # new_logs[k] = [np.mean(vals), np.std(vals)]

    l = list(new_logs.items())
    l = sorted(l, key=lambda x: -x[1][0])
    # pprint.pprint(l)
    for k, (win, loss, tie) in l:
        print('%-20s: %.3f, %.3f, %.3f' % (k, win, loss, tie)) #

    # pprint.pprint(l, width=150)
    return new_logs


def parse_log(log_file):
    lines = open(log_file, 'r').readlines()
    if not (lines[-1].startswith('win') and lines[-2].startswith('win')):
        print('%s is not finished' % log_file)
        return None

    win = int(lines[-2].split()[1][:-1])
    tie_loss = int(lines[-2].split()[3][:-1])
    loss = int(lines[-1].split()[1][:-1])
    tie = tie_loss - loss

    return win, loss, tie


def parse_from_root(root):
    logs = {}
    root = os.path.abspath(root)
    for exp in os.listdir(root):
        exp_folder = os.path.join(root, exp)
        if os.path.isdir(exp_folder):
            log_file = os.path.join(exp_folder, 'train.log')
            if os.path.exists(log_file):
                # print(exp, log_file)
                logs[exp] = parse_log(log_file)

    average_across_seed(logs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True)
    args = parser.parse()
    parse_from_root(args.root)
