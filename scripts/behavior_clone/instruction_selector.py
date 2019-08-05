# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

from common_utils import assert_eq


class OneHotSelector(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()

        self.out_dim = out_dim

        self.mlp = nn.Sequential(
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            weight_norm(nn.Linear(hid_dim, out_dim, bias=False), dim=None),
        )

    def compute_loss(self, x, y):
        """

        x: [batch, dim], context/game_state
        y: [batch], target
        """
        valid_mask = y < self.out_dim
        valid_y = y * valid_mask.long()

        logit = self.mlp(x)
        logp = nn.functional.log_softmax(logit, 1)
        logp = logp.gather(1, valid_y.unsqueeze(1)).squeeze(1)
        logp = logp * valid_mask.float()
        loss = -logp
        return loss

    def compute_prob(self, x):
        logit = self.mlp(x)
        prob = nn.functional.softmax(logit, 1)
        return prob


class RnnSelector(nn.Module):
    def __init__(self, encoder, context_dim):
        super().__init__()
        self.encoder = encoder
        self.context_dim = context_dim
        self.inst_proj = nn.Sequential(
            # weight_norm(nn.Linear(context_dim, encoder.out_dim), dim=None),
            weight_norm(nn.Linear(encoder.out_dim, context_dim), dim=None),
            nn.ReLU(),
        )

    def compute_loss(self,
                     pos_inst,
                     pos_inst_len,
                     neg_inst,
                     neg_inst_len,
                     truth_inst,
                     truth_inst_len,
                     context,
                     truth_idx):
        """

        args:
            inst: [num_inst, max_len}
            inst_len: [num_inst]
            context: [batch, context_dim]
            truth: [batch]
        """
        # assert_eq(pos_inst.size(), neg_inst.size())

        pos_logit = self._forward(pos_inst, pos_inst_len, context)
        neg_logit = self._forward(neg_inst, neg_inst_len, context)
        truth_logit = self._forward1d(truth_inst, truth_inst_len, context)
        # pos_logit/neg_logit: [batch, num_inst]
        # truth_logit: [batch]
        truth_mask = (truth_idx < pos_inst.size(0)).float()
        truth_logit = (truth_logit + truth_mask * 1e-9).unsqueeze(1)
        logit = torch.cat([pos_logit, neg_logit, truth_logit], 1)
        logp = nn.functional.log_softmax(logit, 1)

        truth_idx = truth_mask * truth_idx.float() + (1 - truth_mask) * (logp.size(1) - 1)
        logp = logp.gather(1, truth_idx.long().unsqueeze(1)).squeeze(1)

        loss = -logp
        return loss

    def _forward(self, inst, inst_len, context):
        """

        args:
            inst: [num_inst, max_len}
            inst_len: [num_inst]
            context: [batch, context_dim]
        """
        batch = context.size(0)
        num_inst = inst.size(0)
        inst_feat = self.inst_proj(self.encoder(inst, inst_len))
        # inst_feat: [num_inst, encoder.out_dim]
        inst_feat = inst_feat.unsqueeze(0).repeat(batch, 1, 1)
        context = context.unsqueeze(1).repeat(1, num_inst, 1)

        # inst_feat = self.encoder(inst, inst_len).unsqueeze(0).repeat(batch, 1, 1)
        # context = self.inst_proj(context)
        # context = context.unsqueeze(1).repeat(1, num_inst, 1)
        assert(inst_feat.size() == context.size())
        logit = (inst_feat * context).sum(2)
        return logit

    def _forward1d(self, inst, inst_len, context):
        assert_eq(inst.size(0), context.size(0))
        inst_feat = self.inst_proj(self.encoder(inst, inst_len))
        # inst_feat = self.encoder(inst, inst_len)
        # context = self.inst_proj(context)
        logit = (inst_feat * context).sum(1)
        return logit

    def eval_loss(self, inst, inst_len, context, target):
        logp = self.compute_prob(inst, inst_len, context, log=True)
        logp = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        return -logp

    def compute_prob(self, inst, inst_len, context, *, log=False):
        """

        args:
            inst: [num_inst, max_len}
            inst_len: [num_inst]
            context: [batch, context_dim]
        """
        logit = self._forward(inst, inst_len, context)
        logit = logit
        # logit: [batch, num_inst]
        if log:
            return nn.functional.log_softmax(logit, 1)
        else:
            return nn.functional.softmax(logit, 1)
