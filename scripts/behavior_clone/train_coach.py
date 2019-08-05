# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import sys
import argparse
import pprint
from collections import defaultdict
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from coach_dataset import CoachDataset, compute_cache
from rnn_coach import ConvRnnCoach
from onehot_coach import ConvOneHotCoach
from rnn_generator import RnnGenerator

import common_utils


def train(model, device, optimizer, grad_clip, data_loader, epoch):
    assert model.training

    losses = defaultdict(list)
    t = time.time()
    for batch_idx, batch in enumerate(data_loader):
        batch = common_utils.to_device(batch, device)
        optimizer.zero_grad()
        loss, all_losses = model.compute_loss(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        for key, val in all_losses.items():
            losses[key].append(val.item())

    print('train epoch: %d, time: %.2f' % (epoch, time.time() - t))
    for key, val in losses.items():
        print('\t%s: %.5f' % (key, np.mean(val)))

    return np.mean(losses['loss'])


def evaluate(model, device, data_loader, epoch, name, norm_loss):
    assert not model.training

    losses = defaultdict(list)
    t = time.time()
    for batch_idx, batch in enumerate(data_loader):
        batch = common_utils.to_device(batch, device)
        if norm_loss:
            loss, all_losses = model.compute_eval_loss(batch)
        else:
            loss, all_losses = model.compute_loss(batch)

        for key, val in all_losses.items():
            losses[key].append(val.item())

    print('%s epoch: %d, time: %.2f' % (name, epoch, time.time() - t))
    for key, val in losses.items():
        print('\t%s: %.5f' % (key, np.mean(val)))

    return np.mean(losses['loss'])


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

    parser.add_argument('--coach_type', type=str, required=True)

    # optim
    parser.add_argument('--optim', type=str, default='adamax')
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--grad_clip', type=float, default=0.5)

    # data setting
    parser.add_argument('--moving_avg_decay',
                        type=float, default=0.98, help='moving avg for enemy count')
    parser.add_argument('--num_resource_bin', type=int, default=11)
    parser.add_argument('--resource_bin_size', type=int, default=50)
    parser.add_argument('--max_num_prev_cmds', type=int, default=25)
    parser.add_argument('--max_instruction_span',
                        type=int, default=20, help='max for feature "frame_passed"')

    # debug
    parser.add_argument('--dev', action='store_true')

    return parser


def main():
    torch.backends.cudnn.benchmark = True

    parser = common_utils.Parser()
    parser.add_parser('main', get_main_parser())
    parser.add_parser('coach', ConvRnnCoach.get_arg_parser())

    args = parser.parse()
    parser.log()

    options = args['main']

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

    model_args = args['coach']
    if options.coach_type == 'onehot':
        model = ConvOneHotCoach(
            model_args,
            0,
            options.max_instruction_span,
            options.num_resource_bin).to(device)
    elif options.coach_type in ['rnn', 'bow']:
        model = ConvRnnCoach(
            model_args,
            0,
            options.max_instruction_span,
            options.coach_type,
            options.num_resource_bin).to(device)
    elif options.coach_type == 'rnn_gen':
        model = RnnGenerator(
            model_args,
            0,
            options.max_instruction_span,
            options.num_resource_bin).to(device)

    print(model)

    train_dataset = CoachDataset(
        options.train_dataset,
        options.moving_avg_decay,
        options.num_resource_bin,
        options.resource_bin_size,
        options.max_num_prev_cmds,
        model.inst_dict,
        options.max_instruction_span,
    )
    val_dataset = CoachDataset(
        options.val_dataset,
        options.moving_avg_decay,
        options.num_resource_bin,
        options.resource_bin_size,
        options.max_num_prev_cmds,
        model.inst_dict,
        options.max_instruction_span,
    )
    eval_dataset = CoachDataset(
        options.val_dataset,
        options.moving_avg_decay,
        options.num_resource_bin,
        options.resource_bin_size,
        options.max_num_prev_cmds,
        model.inst_dict,
        options.max_instruction_span,
        num_instructions=model.args.num_pos_inst)

    if not options.dev:
        compute_cache(train_dataset)
        compute_cache(val_dataset)
        compute_cache(eval_dataset)

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
        num_workers=1,# if options.dev else 10,
        pin_memory=(options.gpu >= 0))
    val_loader = DataLoader(
        val_dataset,
        options.batch_size,
        shuffle=False,
        num_workers=1,# if options.dev else 10,
        pin_memory=(options.gpu >= 0))
    eval_loader = DataLoader(
        eval_dataset,
        options.batch_size,
        shuffle=False,
        num_workers=1,#0 if options.dev else 10,
        pin_memory=(options.gpu >= 0))

    best_val_nll = float('inf')
    overfit_count = 0
    for epoch in range(1, options.epochs + 1):
        print('==========')
        train(model, device, optimizer, options.grad_clip, train_loader, epoch)
        with torch.no_grad(), common_utils.EvalMode(model):
            val_nll = evaluate(model, device, val_loader, epoch, 'val', False)
            eval_nll = evaluate(model, device, eval_loader, epoch, 'eval', True)

        model_file = os.path.join(options.model_folder, 'checkpoint%d.pt' % epoch)
        print('saving model to', model_file)
        model.save(model_file)

        if val_nll < best_val_nll:
            print('!!!New Best Model')
            overfit_count = 0
            best_val_nll = val_nll
            best_model_file = os.path.join(options.model_folder, 'best_checkpoint.pt')
            print('saving best model to', best_model_file)
            model.save(best_model_file)
        else:
            overfit_count += 1
            if overfit_count == 2:
                break

    print('train DONE')


if __name__ == '__main__':
    main()
