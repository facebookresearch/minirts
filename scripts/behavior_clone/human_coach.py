# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import os
import pprint

from set_path import append_sys_path
append_sys_path()

import torch
import tube
import pytube
import minirts
from executor_wrapper import ExecutorWrapper
from executor import Executor
from common_utils import to_device


def create_game(ai1_option, ai2_option, game_option, *, act_name='act'):
    print('ai1 option:')
    print(ai1_option.info())
    print('ai2 option:')
    print(ai2_option.info())
    print('game option:')
    print(game_option.info())

    act_dc = tube.DataChannel(act_name, 1, -1)
    context = tube.Context()
    g = minirts.RTSGame(game_option)
    bot1 = minirts.CheatExecutorAI(ai1_option, 0, None, act_dc)
    bot2 = minirts.MediumAI(ai2_option, 0, None, minirts.UnitType.INVALID_UNITTYPE, False)
    g.add_bot(bot1)
    g.add_bot(bot2)
    g.add_default_spectator()

    context.push_env_thread(g)
    return context, act_dc


def parse_args():
    parser = argparse.ArgumentParser(description='human coach')
    # parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--seed', type=int, default=1)
    # parser.add_argument('--deterministic', action='store_true')
    # parser.add_argument('--num_thread', type=int, default=1)
    # parser.add_argument('--batchsize', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    # parser.add_argument('--update_per_epoch', type=int, default=200)
    # parser.add_argument('--num_epoch', type=int, default=400)

    root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    default_lua = os.path.join(root, 'game/game_MC/lua')
    # print(default_lua)
    # assert False
    parser.add_argument('--lua_files', type=str, default=default_lua)

    # ai1 option
    parser.add_argument('--frame_skip', type=int, default=50)
    parser.add_argument('--fow', type=int, default=1)
    # parser.add_argument('--t_len', type=int, default=10)
    parser.add_argument('--use_moving_avg', type=int, default=1)
    parser.add_argument('--moving_avg_decay', type=float, default=0.98)
    parser.add_argument('--num_resource_bins', type=int, default=11)
    parser.add_argument('--resource_bin_size', type=int, default=50)
    parser.add_argument('--max_num_units', type=int, default=50)
    parser.add_argument('--num_prev_cmds', type=int, default=25)
    parser.add_argument('--num_instructions', type=int, default=1,
                        help="not used in human coach, > 0 is sufficient")
    parser.add_argument('--max_raw_chars', type=int, default=200)
    parser.add_argument('--verbose', action='store_true')

    # game option
    parser.add_argument('--max_tick', type=int, default=int(2e5))
    parser.add_argument('--no_terrain', action='store_true')
    parser.add_argument('--resource', type=int, default=500)
    parser.add_argument('--resource_dist', type=int, default=4)
    parser.add_argument('--fair', type=int, default=0)
    parser.add_argument('--save_replay_freq', type=int, default=0)
    parser.add_argument('--save_replay_per_games', type=int, default=1)

    # model
    parser.add_argument(
        '--model_path',
        type=str,
        default='../../pretrained_models/executor_rnn.pt'
    )

    args = parser.parse_args()

    args.model_path = os.path.abspath(args.model_path)
    if not os.path.exists(args.model_path):
        print('cannot find model at:', args.model_path)
        assert False

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
    game_option.num_games_per_thread = 1
    # !!! this is important
    game_option.max_num_units_per_player = args.max_num_units
    return game_option


def get_ai_option(args):
    ai1_option = minirts.AIOption()
    ai1_option.t_len = 1;
    ai1_option.fs = args.frame_skip
    ai1_option.fow = args.fow
    ai1_option.use_moving_avg = args.use_moving_avg
    ai1_option.moving_avg_decay = args.moving_avg_decay
    ai1_option.num_resource_bins = args.num_resource_bins
    ai1_option.resource_bin_size = args.resource_bin_size
    ai1_option.max_num_units = args.max_num_units
    ai1_option.num_prev_cmds = args.num_prev_cmds
    ai1_option.num_instructions = args.num_instructions
    ai1_option.max_raw_chars = args.max_raw_chars
    ai1_option.verbose = args.verbose

    ai2_option = minirts.AIOption()
    ai2_option.fs = args.frame_skip
    ai2_option.fow = args.fow
    return ai1_option, ai2_option


if __name__ == '__main__':
    args = parse_args()
    print('args:')
    pprint.pprint(vars(args))

    os.environ['LUA_PATH'] = os.path.join(args.lua_files, '?.lua')
    print('lua path:', os.environ['LUA_PATH'])

    game_option = get_game_option(args)
    ai1_option, ai2_option = get_ai_option(args)
    context, act_dc = create_game(ai1_option, ai2_option, game_option)

    device = torch.device('cuda:%d' % args.gpu)
    executor = Executor.load(args.model_path).to(device)
    print('top 500 insts')
    for inst  in executor.inst_dict._idx2inst[:500]:
        print(inst)
    executor_wrapper = ExecutorWrapper(
        None, executor, args.num_instructions, args.max_raw_chars, False)
    executor_wrapper.train(False)

    context.start()
    dc = pytube.DataChannelManager([act_dc])
    while not context.terminated():
        data = dc.get_input()['act']
        data = to_device(data, device)
        # import IPython
        # IPython.embed()
        reply = executor_wrapper.forward(data)

        # reply = {key : reply[key].detach().cpu() for key in reply}
        dc.set_reply('act', reply)
        print('===end of a step===')

    # import IPython
    # IPython.embed()
