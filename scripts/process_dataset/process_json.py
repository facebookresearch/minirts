# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os
import json
import tempfile
from collections import defaultdict

import multiprocessing
import time
from tqdm import tqdm

from process_game import process_game
from format_json import format_json
from utils import get_all_files


def process_all(root, output):
    assert formatted or compact

    src_files = get_all_files(root, '.json')

    for f in tqdm(src_files):
        game = json.load(open(f, 'r'))
        game = process_game(game, f)

        dest_file = f.replace(root, output)
        dirname = os.path.dirname(dest_file)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        temp_json = tempfile.TemporaryFile('r+')
        json.dump(game, temp_json, indent=4)
        temp_json.seek(0)
        formatted = format_json(temp_json)
        with open(dest_file, 'w') as f:
            f.write(formatted)

    # t = time.time()
    # print('start processing')
    # pool = multiprocessing.Pool(60)
    # pool.map(_process_one, src_files)
    # print('time taken: %.2f' % time - t)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True)
    parser.add_argument('--dest', type=str, default=None)

    args = parser.parse_args()
    if args.dest is None:
        args.dest = args.src + '_processed'

    process_all(args.src, args.dest, True, False)
