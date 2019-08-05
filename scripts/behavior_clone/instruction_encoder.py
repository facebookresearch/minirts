# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class ZeroInstructionEncoder(nn.Module):
    def __init__(self, out_size):
        super().__init__()
        self.out_dim = out_size
        self.emb = nn.Embedding(1, out_size, padding_idx=0)

    def forward(self, x, sizes):
        x.fill_(0)
        emb = self.emb(x).sum(1)
        return emb


class TopNInstructionEncoder(nn.Module):
    def __init__(self, num_inst, emb_dropout, out_size):
        """
        padding_idx will be num_inst
        """
        super().__init__()
        self.num_inst = num_inst
        self.out_dim = out_size
        self.emb = nn.Embedding(num_inst+1, out_size, padding_idx=num_inst)

    def forward(self, x, _):
        """
        if x >= num_inst, the embedding will be 0
        """
        assert x.dim() == 2
        x  = x.clamp(max=self.num_inst)
        return self.emb(x)


class MeanBOWInstructionEncoder(nn.Module):
    def __init__(self, dict_size, emb_size, emb_dropout, padding_idx, *, emb=None):
        super().__init__()
        self.out_dim = emb_size
        if emb is None:
            self.emb = nn.Embedding(dict_size, emb_size, padding_idx=padding_idx)
        else:
            self.emb = emb
        self.dropout = nn.Dropout(emb_dropout)

    def forward(self, x, sizes):
        e = self.emb(x)
        e = self.dropout(e)
        bow = e.mean(dim=1)
        return bow


class LSTMInstructionEncoder(nn.Module):
    def __init__(self, dict_size, emb_size, emb_dropout, out_size, padding_idx):
        super().__init__()
        self.out_dim = out_size
        self.emb = nn.Embedding(dict_size, emb_size, padding_idx=padding_idx)
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.rnn = nn.LSTM(emb_size, out_size, batch_first=True)

    def forward(self, x, sizes):
        # assert(x.dim(), 3)
        e = self.emb(x)
        e = self.emb_dropout(e)
        h = torch.zeros(1, x.size(0), self.out_dim).to(x.device)
        c = torch.zeros(1, x.size(0), self.out_dim).to(x.device)
        hs, _ = self.rnn(e, (h, c))
        mask = (sizes > 0).long()
        indexes = (sizes - mask).unsqueeze(1).unsqueeze(2)
        indexes = indexes.expand(indexes.size(0), 1, hs.size(2))
        h = hs.gather(1, indexes).squeeze(1)
        # h: [batch, out_size]
        h = h * mask.float().unsqueeze(1)
        return h


class GRUInstructionEncoder(nn.Module):
    def __init__(self, dict_size, emb_size, emb_dropout, out_size, padding_idx):
        super().__init__()
        self.out_dim = out_size
        self.emb = nn.Embedding(dict_size, emb_size, padding_idx=padding_idx)
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.rnn = nn.GRU(emb_size, out_size, batch_first=True)

    def forward(self, x, sizes):
        # assert(x.dim(), 3)
        e = self.emb(x)
        e = self.emb_dropout(e)
        h = torch.zeros(1, x.size(0), self.out_dim).to(x.device)
        hs, _ = self.rnn(e, h)
        mask = (sizes > 0).long()
        indexes = (sizes - mask).unsqueeze(1).unsqueeze(2)
        indexes = indexes.expand(indexes.size(0), 1, hs.size(2))
        h = hs.gather(1, indexes).squeeze(1)
        # h: [batch, out_size]
        h = h * mask.float().unsqueeze(1)
        return h


class HistInstructionEncoder(nn.Module):
    def __init__(self, encoder, num_inst, pos_dim, army_dim):
        super().__init__()

        self.out_dim = pos_dim * 2 + encoder.out_dim
        self.encoder = encoder
        self.emb = nn.Embedding(num_inst, pos_dim)
        self.max_hist_diff = 10
        self.diff_emb = nn.Embedding(self.max_hist_diff + 1, pos_dim)
        self.proj = nn.Sequential(
            weight_norm(nn.Linear(army_dim, self.out_dim), dim=None),
            nn.ReLU(),
        )

    def forward(self, army_feat, hist_inst, hist_inst_len, hist_inst_diff):
        """
        army_feat: [batch, num_unit, feat_dim]
        hist_inst: [batch, num_hist, sentence_len]
        """
        batch, num_hist = hist_inst.size()[:2]
        hist_inst = hist_inst.view(batch * num_hist, -1)
        hist_inst_len = hist_inst_len.view(batch * num_hist)
        hist_feat = self.encoder(hist_inst, hist_inst_len)
        hist_feat = hist_feat.view(batch, num_hist, -1)
        # hist_feat: [batch, num_hist, inst_dim]
        pos_emb = self.emb.weight.unsqueeze(0).repeat(batch, 1, 1)
        hist_inst_diff = hist_inst_diff.clamp(max=self.max_hist_diff)
        diff_emb = self.diff_emb(hist_inst_diff)
        # print(diff_emb.size(), pos_emb.size(), hist_feat.size())

        hist_feat = torch.cat([hist_feat, pos_emb, diff_emb], 2)
        # print('hist feat', hist_feat.size())
        sum_inst = hist_feat[:, -1]
        # print('sum_inst:', sum_inst.size())

        num_army = army_feat.size(1)
        hist_feat = hist_feat.unsqueeze(1).repeat(1, num_army, 1, 1)

        proj_army = self.proj(army_feat)
        proj_army = proj_army.unsqueeze(2).repeat(1, 1, num_hist, 1)

        # print(hist_feat.size())
        dot_prod = (proj_army * hist_feat).sum(3)
        # dot_prod: [batch, num_army, num_hist]
        att_score = nn.functional.softmax(dot_prod, 2)
        # att_score: [batch, num_army, num_hist]
        # hist_feat: [batch, num_army, num_hist, hist_dim]
        inst_feat = (att_score.unsqueeze(3) * hist_feat).sum(2)
        # inst_feat: [batch, num_army, hist_dim]
        return inst_feat, sum_inst


# ENCODERS = {
#     # 'random': RandomInstructionEncoder,
#     # 'constant': ConstantInstructionEncoder,
#     'topn': TopNInstructionEncoder,
#     # 'bow': BOWInstructionEncoder,
#     'lstm': LSTMInstructionEncoder,
#     'gru': GRUInstructionEncoder,
#     'zero': ZeroInstructionEncoder,
# }


# def get_instruction_encoder(name):
#     assert name in ENCODERS
#     return ENCODERS[name]


def is_word_based(name):
    return name in ('random', 'bow', 'lstm', 'gru', 'zero')
