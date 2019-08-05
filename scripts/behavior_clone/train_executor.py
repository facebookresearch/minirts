# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import sys
import pprint
import argparse
from collections import defaultdict
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import BehaviorCloneDataset, merge_max_units
from executor import Executor
from instruction_encoder import is_word_based
import common_utils


def train(model, device, optimizer, grad_clip, data_loader, epoch, stat):
    assert model.training

    # losses = defaultdict(list)
    for batch_idx, batch in enumerate(data_loader):
        batch = common_utils.to_device(batch, device)
        optimizer.zero_grad()
        # loss, all_losses = model.compute_loss(batch)
        loss, all_losses = model(batch)
        # print(loss.mean())
        # print(all_losses)
        loss = loss.mean()

        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        for key, val in all_losses.items():
            v = val.mean().item()
            stat[key].feed(v)

            # losses[key].append(val.item())

    # for key, val in losses.items():
    #     print('\t%s: %.5f' % (key, np.mean(val)))
    # return np.mean(losses['loss'])
    return stat['loss'].mean()


def evaluate(model, device, data_loader, epoch, stat):
    assert not model.training

    # losses = defaultdict(list)
    for batch_idx, batch in enumerate(data_loader):
        batch = common_utils.to_device(batch, device)
        loss, all_losses = model(batch)
        # for key, val in all_losses.items():
        #     losses[key].append(val.item())
        for key, val in all_losses.items():
            stat[key].feed(val.mean().item())

    # print('eval:')
    #     print('\t%s: %.5f' % (key, np.mean(val)))

    # return np.mean(losses['loss'])
    return stat['loss'].mean()


def get_main_parser():
    parser = argparse.ArgumentParser()

    # train config
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--train_dataset', type=str, required=True)
    parser.add_argument('--val_dataset', type=str, required=True)
    parser.add_argument('--model_folder', type=str, required=True)
    # 'folder to save model', 'model-test')

    # optim
    parser.add_argument('--optim', type=str, default='adamax')
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--grad_clip', type=float, default=0.5)

    # debug
    parser.add_argument('--dev', action='store_true')

    # data config
    parser.add_argument('--num_resource_bin', type=int, default=11)
    parser.add_argument('--resource_bin_size', type=int, default=50)
    parser.add_argument('--max_num_prev_cmds', type=int, default=25)

    return parser


def main():
    torch.backends.cudnn.benchmark = True

    parser = common_utils.Parser()
    parser.add_parser('main', get_main_parser())
    parser.add_parser('executor', Executor.get_arg_parser())
    args = parser.parse()
    parser.log()

    options = args['main']
    # print('Args:\n%s\n' % pprint.pformat(vars(options)))

    # option_map = parse_args()
    # options = option_map.getOptions()

    if not os.path.exists(options.model_folder):
        os.makedirs(options.model_folder)
    logger_path = os.path.join(options.model_folder, 'train.log')
    if not options.dev:
        sys.stdout = common_utils.Logger(logger_path)

    if options.dev:
        options.train_dataset = options.train_dataset.replace('train.', 'dev.')
        options.val_dataset = options.val_dataset.replace('val.', 'dev.')

    print('Args:\n%s\n' % pprint.pformat(vars(options)))

    if options.gpu < 0:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:%d' % options.gpu)

    common_utils.set_all_seeds(options.seed)

    model = Executor(args['executor'], options.num_resource_bin).to(device)
    inst_dict = model.inst_dict
    print(model)
    # model = nn.DataParallel(model, [0, 1])

    train_dataset = BehaviorCloneDataset(
        options.train_dataset,
        options.num_resource_bin,
        options.resource_bin_size,
        options.max_num_prev_cmds,
        inst_dict=inst_dict,
        word_based=is_word_based(args['executor'].inst_encoder_type))
    val_dataset = BehaviorCloneDataset(
        options.val_dataset,
        options.num_resource_bin,
        options.resource_bin_size,
        options.max_num_prev_cmds,
        inst_dict=inst_dict,
        word_based=is_word_based(args['executor'].inst_encoder_type))

    if options.optim == 'adamax':
        optimizer = torch.optim.Adamax(
            model.parameters(),
            lr=options.lr,
            betas=(options.beta1, options.beta2))
    elif options.optim == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=options.lr,
            betas=(options.beta1, options.beta2))
    else:
        assert False, 'not supported'

    train_loader = DataLoader(
        train_dataset,
        options.batch_size,
        shuffle=True,
        num_workers=20, # if options.dev else 20,
        pin_memory=(options.gpu >= 0))
    val_loader = DataLoader(
        val_dataset,
        options.batch_size,
        shuffle=False,
        num_workers=20, # if options.dev else 20,
        pin_memory=(options.gpu >= 0))

    best_eval_nll = float('inf')
    overfit_count = 0

    train_stat = common_utils.MultiCounter(os.path.join(options.model_folder, 'train'))
    eval_stat = common_utils.MultiCounter(os.path.join(options.model_folder, 'eval'))
    for epoch in range(1, options.epochs + 1):
        train_stat.start_timer()
        train(model, device, optimizer, options.grad_clip, train_loader, epoch, train_stat)
        train_stat.summary(epoch)
        train_stat.reset()

        with torch.no_grad(), common_utils.EvalMode(model):
            eval_stat.start_timer()
            eval_nll = evaluate(model, device, val_loader, epoch, eval_stat)
            eval_stat.summary(epoch)
            eval_stat.reset()

        model_file = os.path.join(options.model_folder, 'checkpoint%d.pt' % epoch)
        print('saving model to', model_file)
        if isinstance(model, nn.DataParallel):
            model.module.save(model_file)
        else:
            model.save(model_file)

        if eval_nll < best_eval_nll:
            print('!!!New Best Model')
            overfit_count = 0
            best_eval_nll = eval_nll
            best_model_file = os.path.join(options.model_folder, 'best_checkpoint.pt')
            print('saving best model to', best_model_file)
            if isinstance(model, nn.DataParallel):
                model.module.save(best_model_file)
            else:
                model.save(best_model_file)
        else:
            overfit_count += 1
            if overfit_count == 2:
                break

    print('train DONE')


if __name__ == '__main__':
    main()
