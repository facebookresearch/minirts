# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""A wrapper class for some optimization related helpers"""
from contextlib import contextmanager
import torch
from torch import nn


# @contextmanager
# def eval_mode_no_grad(model):
#     in_train_mode = model.training
#     with torch.no_grad():
#         if in_train_mode:
#             model.train(False)
#         yield

#     if in_train_mode:
#         model.train(True)


class Optim:
    def __init__(self, model, optim_cls, optim_args, max_grad_norm):
        self.model = model
        self.optim = optim_cls(self.model.parameters(), **optim_args)
        self.max_grad_norm = max_grad_norm

    def assert_zero_grad(self):
        for p in self.model.parameters():
            if p.grad is not None:
                assert(p.grad.sum().item() == 0)

    def zero_grad(self):
        self.optim.zero_grad()

    def step(self, stat):
        if self.max_grad_norm is not None:
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm)
            if stat is not None:
                # print('grad_norm:', grad_norm)
                stat['grad_norm'].feed(grad_norm)

        self.optim.step()
        self.optim.zero_grad()
        if self.max_grad_norm is not None:
            return grad_norm
