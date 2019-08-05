# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import random
import numpy as np
import torch


def to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device).detach()
    elif isinstance(batch, dict):
        return {key : to_device(batch[key], device) for key in batch}
    else:
        assert False, 'unsupported type: %s' % type(batch)


def flatten_first2dim(batch):
    if isinstance(batch, torch.Tensor):
        size = batch.size()[2:]
        batch = batch.view(-1, *size)
        return batch
    elif isinstance(batch, dict):
        return {key : flatten_first2dim(batch[key]) for key in batch}
    else:
        assert False, 'unsupported type: %s' % type(batch)


def _tensor_slice(t, dim, b, e):
    if dim == 0:
        return t[b:e]
    elif dim == 1:
        return t[:, b:e]
    elif dim == 2:
        return t[:, :, b:e]
    else:
        raise ValueError('unsupported %d in tensor_slice' % dim)

def tensor_slice(t, dim, b, e):
    if isinstance(t, dict):
        return {key : tensor_slice(t[key], dim, b, e) for key in t}
    elif isinstance(t, torch.Tensor):
        return _tensor_slice(t, dim, b, e).contiguous()
    else:
        assert False, 'Error: unsupported type: %s' % (type(t))


def tensor_index(t, dim, i):
    if isinstance(t, dict):
        return {key : tensor_index(t[key], dim, i) for key in t}
    elif isinstance(t, torch.Tensor):
        return _tensor_slice(t, dim, i, i + 1).squeeze(dim).contiguous()
    else:
        assert False, 'Error: unsupported type: %s' % (type(t))


def one_hot(x, n):
    assert x.dim() == 2 and x.size(1) == 1
    one_hot_x = torch.zeros(x.size(0), n, device=x.device)
    one_hot_x.scatter_(1, x, 1)
    return one_hot_x


def set_all_seeds(rand_seed):
    random.seed(rand_seed)
    np.random.seed(rand_seed + 1)
    torch.manual_seed(rand_seed + 2)
    torch.cuda.manual_seed(rand_seed + 3)


class EvalMode:
    def __init__(self, module):
        self.module = module
        self.training = self.module.training

    def __enter__(self):
        if self.training:
            self.module.train(False)
        return self.module

    def __exit__(self, type_, value, traceback):
        if self.training:
            self.module.train(True)
