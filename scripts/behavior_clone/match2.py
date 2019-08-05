# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os
import sys
import pprint

from set_path import append_sys_path
append_sys_path()

import torch
import tube
from pytube import DataChannelManager
import minirts
from rnn_coach import ConvRnnCoach
from onehot_coach import ConvOneHotCoach
from rnn_generator import RnnGenerator

from executor_wrapper import ExecutorWrapper
from executor import Executor
from common_utils import to_device, ResultStat, Logger
from best_models import best_executors, best_coaches


def create_game(num_games, ai1_option, ai2_option, game_option, *, act_name='act'):
    print('ai1 option:')
    print(ai1_option.info())
    print('ai2 option:')
    print(ai2_option.info())
    print('game option:')
    print(game_option.info())

    batchsize = min(32, max(num_games // 2, 1))
    act1_dc = tube.DataChannel(act_name+'1', batchsize, 1)
    act2_dc = tube.DataChannel(act_name+'2', batchsize, 1)
    context = tube.Context()
    idx2utype = [
        minirts.UnitType.SPEARMAN,
        minirts.UnitType.SWORDMAN,
        minirts.UnitType.CAVALRY,
        minirts.UnitType.DRAGON,
        minirts.UnitType.ARCHER,
    ]

    for i in range(num_games):
        g_option = minirts.RTSGameOption(game_option)
        g_option.seed = game_option.seed + i
        if game_option.save_replay_prefix:
            g_option.save_replay_prefix = game_option.save_replay_prefix + str(i)

        g = minirts.RTSGame(g_option)
        bot1 = minirts.CheatExecutorAI(ai1_option, 0, None, act1_dc)
        bot2 = minirts.CheatExecutorAI(ai2_option, 0, None, act2_dc)
        # utype = idx2utype[i % len(idx2utype)]
        # bot2 = minirts.MediumAI(ai2_option, 0, None, utype, False)
        g.add_bot(bot1)
        g.add_bot(bot2)
        context.push_env_thread(g)

    return context, act1_dc, act2_dc


def parse_args():
    parser = argparse.ArgumentParser(description='human coach')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_thread', type=int, default=1)
    parser.add_argument('--game_per_thread', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)

    root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    default_lua = os.path.join(root, 'game/game_MC/lua')
    parser.add_argument('--lua_files', type=str, default=default_lua)

    # ai1 option
    parser.add_argument('--frame_skip', type=int, default=50)
    parser.add_argument('--fow', type=int, default=1)
    parser.add_argument('--use_moving_avg', type=int, default=1)
    parser.add_argument('--moving_avg_decay', type=float, default=0.98)
    parser.add_argument('--num_resource_bins', type=int, default=11)
    parser.add_argument('--resource_bin_size', type=int, default=50)
    parser.add_argument('--max_num_units', type=int, default=50)
    parser.add_argument('--num_prev_cmds', type=int, default=25)
    # TOOD: add max instruction span

    parser.add_argument('--max_raw_chars', type=int, default=200)
    parser.add_argument('--verbose', action='store_true')

    parser.add_argument('--inst_mode', type=str, default='full') # can be full/good/better

    # game option
    parser.add_argument('--max_tick', type=int, default=int(2e4))
    parser.add_argument('--no_terrain', action='store_true')
    parser.add_argument('--resource', type=int, default=500)
    parser.add_argument('--resource_dist', type=int, default=4)
    parser.add_argument('--fair', type=int, default=0)
    parser.add_argument('--save_replay_freq', type=int, default=0)
    parser.add_argument('--save_replay_per_games', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='matches2/dev')

    # full vision
    parser.add_argument('--cheat', type=int, default=0)

    parser.add_argument('--coach1', type=str, default='')
    parser.add_argument('--executor1', type=str, default='')

    parser.add_argument('--coach2', type=str, default='')
    parser.add_argument('--executor2', type=str, default='')

    args = parser.parse_args()
    args.coach1 = best_coaches[args.coach1]
    args.executor1 = best_executors[args.executor1]
    args.coach2 = best_coaches[args.coach2]
    args.executor2 = best_executors[args.executor2]

    return args


def get_game_option(args):
    game_option = minirts.RTSGameOption()
    game_option.seed = args.seed
    game_option.max_tick = args.max_tick
    game_option.no_terrain = args.no_terrain
    game_option.resource = args.resource
    game_option.resource_dist = args.resource_dist
    game_option.fair = args.fair
    game_option.save_replay_freq = args.save_replay_freq
    game_option.save_replay_per_games = args.save_replay_per_games
    game_option.lua_files = args.lua_files
    game_option.num_games_per_thread = args.game_per_thread
    # !!! this is important
    game_option.max_num_units_per_player = args.max_num_units

    save_dir = os.path.abspath(args.save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    game_option.save_replay_prefix = save_dir + '/'

    return game_option


def get_ai_options(args, num_instructions):
    options = []
    for i in range(2):
        ai_option = minirts.AIOption()
        ai_option.t_len = 1
        ai_option.fs = args.frame_skip
        ai_option.fow = args.fow
        ai_option.use_moving_avg = args.use_moving_avg
        ai_option.moving_avg_decay = args.moving_avg_decay
        ai_option.num_resource_bins = args.num_resource_bins
        ai_option.resource_bin_size = args.resource_bin_size
        ai_option.max_num_units = args.max_num_units
        ai_option.num_prev_cmds = args.num_prev_cmds
        ai_option.num_instructions = num_instructions[i]
        ai_option.max_raw_chars = args.max_raw_chars
        ai_option.verbose = args.verbose
        options.append(ai_option)

    return options[0], options[1]


def load_model(coach_path, model_path, args):
    if 'onehot' in coach_path:
        coach = ConvOneHotCoach.load(coach_path).to(device)
    elif 'gen' in coach_path:
        coach = RnnGenerator.load(coach_path).to(device)
    else:
        coach = ConvRnnCoach.load(coach_path).to(device)
    coach.max_raw_chars = args.max_raw_chars
    executor = Executor.load(model_path).to(device)
    executor_wrapper = ExecutorWrapper(
        coach, executor, coach.num_instructions, args.max_raw_chars, args.cheat, args.inst_mode)
    executor_wrapper.train(False)
    return executor_wrapper


if __name__ == '__main__':
    args = parse_args()
    print('args:')
    pprint.pprint(vars(args))

    os.environ['LUA_PATH'] = os.path.join(args.lua_files, '?.lua')
    print('lua path:', os.environ['LUA_PATH'])

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    logger_path = os.path.join(args.save_dir, 'train.log')
    sys.stdout = Logger(logger_path)

    device = torch.device('cuda:%d' % args.gpu)

    model1 = load_model(args.coach1, args.executor1, args)
    model2 = load_model(args.coach2, args.executor2, args)

    game_option = get_game_option(args)
    ai1_option, ai2_option = get_ai_options(
        args, [model1.coach.num_instructions, model2.coach.num_instructions])

    context, act1_dc, act2_dc = create_game(
        args.num_thread, ai1_option, ai2_option, game_option)
    context.start()
    dc = DataChannelManager([act1_dc, act2_dc])

    result1 = ResultStat('reward', None)
    result2 = ResultStat('reward', None)
    i = 0
    while not context.terminated():
        i += 1
        if i % 1000 == 0:
            print('%d, progress agent1: win %d, loss %d' % (i, result1.win, result1.loss))

        data = dc.get_input(max_timeout_s=1)
        if len(data) == 0:
            continue
        for key in data:
            # print(key)
            batch = to_device(data[key], device)
            if key == 'act1':
                result1.feed(batch)
                with torch.no_grad():
                    reply = model1.forward(batch)
            elif key == 'act2':
                result2.feed(batch)
                with torch.no_grad():
                    reply = model2.forward(batch)
            else:
                assert False
            dc.set_reply(key, reply)

    print(result1.log(0))
    print(result2.log(0))
    dc.terminate()
