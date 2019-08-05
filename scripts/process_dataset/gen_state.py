# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import sys
import argparse
import subprocess
from tqdm import tqdm
from common_utils import Logger

from utils import get_all_files


def generate_states(replays, args, src_root, dest_root):
    processes = []
    print('queueing jobs...')
    for replay in tqdm(replays):
        json_prefix = replay.replace(src_root, dest_root)
        # print(json_prefix)
        # print(src_root, dest_root)
        # assert False
        dirname = os.path.dirname(json_prefix)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        if args.human:
            lua_path = replay + '.lua'
        else:
            lua_path = os.path.dirname(args.binary)
        cmd = [
            'LUA_PATH=%s' % (os.path.join(lua_path, '?.lua')),
            args.binary,
            'teamreplay' if args.human else 'replay',
            '--load_replay %s' % replay,
            '--players "%s;%s"' % (args.player1, args.player2),
            '--lua_files %s' % lua_path,
            '--max_tick %d' % args.max_tick,
            '--save_replay_prefix %s' % json_prefix
        ]
        cmd = ' '.join(cmd)
        p = subprocess.Popen(
            [cmd],
            shell=True,
        # )
        # p.wait()
        # print(p.poll())
            stdout=open(os.devnull, 'w'), #subprocess.PIPE,
            stderr=open(os.devnull, 'w'), #subprocess.PIPE
        )
        processes.append((replay, p))
        # assert False

    print('submited jobs: %d' % len(processes))
    print('processing jobs...')

    bad_replays = []
    for replay, p in tqdm(processes):
        p.wait()
        if p.poll() != 0:
            bad_replays.append(replay)
    return bad_replays


def main():
    parser = argparse.ArgumentParser(description='generate states from replays')
    parser.add_argument('--binary', type=str, default='../../build/minirts-backend')
    parser.add_argument('--replays-root', type=str)
    parser.add_argument('--output-root', type=str)
    parser.add_argument('--replay-file-extension', type=str, default='.rep')
    # configs for generating jsons
    parser.add_argument('--human', action='store_true', default=False)
    parser.add_argument('--player1', type=str, default='dummy,fs=50')
    parser.add_argument('--player2', type=str, default='dummy,fs=50')
    parser.add_argument('--max-tick', type=int, default=40000)
    args = parser.parse_args()

    args.binary = os.path.abspath(args.binary)
    if not os.path.exists(args.binary):
        print('cannot find binary at:', args.binary)
        assert False

    logger_path = os.path.join(args.output_root, 'config')
    sys.stdout = Logger(logger_path)

    replays = get_all_files(
        args.replays_root,
        args.replay_file_extension)

    bad_replays = generate_states(replays, args, args.replays_root, args.output_root)
    print('number of corrupted replays: %d' % len(bad_replays))
    for replay in bad_replays:
        print(replay)


if __name__ == '__main__':
    main()
