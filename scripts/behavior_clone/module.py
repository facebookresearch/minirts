# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

import reducer

from common_utils import assert_eq
import common_utils.global_consts as gc


class DeepSet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = weight_norm(nn.Linear(in_dim, out_dim), dim=None)
        self.max_pool = reducer.MaxPooling()

    def forward(self, feat, num_unit):
        feat_max = self.max_pool(feat, num_unit).unsqueeze(1)
        feat = feat - feat_max
        feat = self.linear(feat)
        return nn.functional.relu(feat)


class UnitEmbedding(nn.Module):
    def __init__(self, num_utype, num_ufeat, emb_dim):
        super().__init__()

        self.num_utype = num_utype
        self.num_ufeat = num_ufeat
        self.emb_dim = emb_dim
        self.out_dim = emb_dim * 2

        self.utype_emb = nn.Embedding(num_utype, emb_dim)
        self.ufeat_emb = weight_norm(nn.Linear(num_ufeat, emb_dim), dim=None)

    def forward(self, utype, ufeat):
        """take unit_type and unit_feature, produce unit_repr

        utype: [batch, num_padded_unit],
          0 <= utype < num_utype, padding unit should have a valid unit_type
        ufeat: [batch, num_padded_unit, num_ufeat]
        """
        utype_emb = self.utype_emb(utype)
        ufeat_emb = self.ufeat_emb(ufeat)
        return torch.cat([utype_emb, ufeat_emb], 2)


class UnitTypeHpEmbedding(nn.Module):
    def __init__(self, num_utype, num_hp_bins, emb_dim):
        super().__init__()

        self.num_utype = num_utype
        self.num_hp_bins = num_hp_bins
        self.emb_dim = emb_dim
        self.out_dim = emb_dim * 2

        self.utype_emb = nn.Embedding(num_utype, emb_dim)
        # self.etype_emb = nn.Embedding(num_utype, emb_dim)
        self.hp_emb = nn.Embedding(num_hp_bins, emb_dim)

    def forward(self, utype, hp):
        """take unit_type and unit_feature, produce unit_repr

        utype: [batch, num_padded_unit],
          0 <= utype < num_utype, padding unit should have a valid unit_type
        ufeat: [batch, num_padded_unit, num_ufeat]
        """
        type_emb = self.utype_emb(utype)
        hp = (hp * (self.num_hp_bins - 1)).long()
        hp_emb = self.hp_emb(hp)
        unit_emb = torch.cat([type_emb, hp_emb], 2)
        return unit_emb


class CmdEmbedding(nn.Module):
    def __init__(self, num_ctype, num_utype, emb_dim):
        super().__init__()

        self.num_ctype = num_ctype
        self.num_utype = num_ctype
        self.emb_dim = emb_dim
        self.out_dim = emb_dim * 2

        self.ctype_emb = nn.Embedding(num_ctype, emb_dim)
        self.utype_emb = nn.Embedding(num_utype, emb_dim)

    def forward(self, ctype, utype):
        """take cmd_type, produce cmd_repr

        ctype: [batch, num_padded_unit],
          0 <= ctype < num_ctype, padding cmd should have a valid unit_type
        """
        ctype_emb = self.ctype_emb(ctype)
        utype_emb = self.utype_emb(utype)
        # print('>>>>', ctype.size())
        return torch.cat([ctype_emb, utype_emb], 2)


class MultCmdEmbedding(nn.Module):
    def __init__(self, num_ctype, num_utype, emb_dim):
        super().__init__()

        self.num_ctype = num_ctype
        self.num_utype = num_utype
        self.emb_dim = emb_dim

        self.ctype_emb = nn.Embedding(num_ctype, emb_dim)
        self.utype_emb = nn.Embedding(num_utype, emb_dim)
        self.ccont_emb = nn.Embedding(2, emb_dim)

    def forward(self, ctype, ccont, utype, num_unit):
        ctype_emb = self.ctype_emb(ctype)
        ccont_emb = self.ccont_emb(ccont)
        utype_emb = self.utype_emb(utype)

        not_cont = (1 - ccont.float()).unsqueeze(2)
        emb = ctype_emb * not_cont * utype_emb

        # emb = ctype_emb * ccont_emb * utype_emb
        emb = reducer.SumGlobReducer._sum(emb, num_unit)
        # print('reduced unit emb:', emb.size())
        return emb


class PrevCmdEmbedding(nn.Module):
    def __init__(self, num_ctype, emb_dim):
        super().__init__()

        self.num_ctype = num_ctype
        self.emb_dim = emb_dim
        self.out_dim = emb_dim

        # TODO: should use mask here
        self.ctype_emb = nn.Embedding(num_ctype, emb_dim, padding_idx=0)

    def forward(self, prev_cmd, num_cmd):
        """take cmd_type, produce cmd_repr

        prev_cmd: [batch, num_padded_unit, num_padded_prev_cmds],
        num_cmd: [batch, num_padded_unit]
          0 <= ctype < num_ctype, padding cmd should have a valid unit_type
        """
        assert_eq(prev_cmd.dim(), 3)

        ctype_emb = self.ctype_emb(prev_cmd)
        # ctype_emb [batch, num_padded_unit, num_padded_prev_cmds, emb_dim]
        ctype_emb = ctype_emb.sum(2)
        return ctype_emb


class PrevCmdRnn(nn.Module):
    def __init__(self, num_ctype, emb_dim):
        super().__init__()

        self.num_ctype = num_ctype
        self.emb_dim = emb_dim
        self.out_dim = emb_dim

        self.emb = nn.Embedding(num_ctype, emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(emb_dim, self.out_dim, batch_first=True)

    def forward(self, prev_cmd, num_cmd):
        """
        prev_cmd: [batch, num_unit, num_padded_prev_cmds]
        """
        e = self.emb(prev_cmd)
        # e: [batch, num_unit, num_padded_prev_cmds, emb_dim]
        batch, num_unit, num_cmd, emb_dim = e.size()
        e = e.view(batch * num_unit, num_cmd, emb_dim)
        h, _ = self.rnn(e)
        h = h[:, -1]
        h = h.view(batch, num_unit, self.out_dim)
        return h


# prev_cmd = (torch.rand(4, 3, 5) * 5).long()
# prev_rnn = PrevCmdRnn(5, 12)


class AdvancedCmdEmbedding(nn.Module):
    def __init__(self,
                 num_cmd_type,
                 num_target_type,
                 target_feat_dim,
                 attribute_dim):
        """
        num_cmd_type: num of cmd types
        cmd_type_emb_dim: cmd emb output dim
        target_type_emb: emb function for target unit type, shared with UnitEmbedding
        target_type_emb_dim: unit type emb output dim
        xy_feat_dim: output dim for xy-net
        target_feat_dim: feat dim of enemy/resource encoder

        """
        super().__init__()

        self.num_cmd_type = num_cmd_type
        self.num_target_type = num_target_type
        self.target_feat_dim = target_feat_dim

        self.attribute_dim = attribute_dim
        self.out_dim = 4 * attribute_dim

        self.cmd_type_emb = nn.Embedding(num_cmd_type, attribute_dim)
        self.target_type_emb = nn.Embedding(num_target_type, attribute_dim)
        self.xy_net = nn.Sequential(
            weight_norm(nn.Linear(2, attribute_dim), dim=None),
        )
        self.enemy_feat_net = nn.Sequential(
            weight_norm(nn.Linear(target_feat_dim, attribute_dim), dim=None),
        )
        self.resource_feat_net = nn.Sequential(
            weight_norm(nn.Linear(target_feat_dim, attribute_dim), dim=None),
        )

        # compute locations in output feature tensor
        self.cmd_type_emb_end = attribute_dim
        self.target_type_emb_end = 2 * attribute_dim
        self.xy_feat_end = 3 * attribute_dim
        self.target_feat_end = 4 * attribute_dim
        assert_eq(self.target_feat_end, self.out_dim)

    @staticmethod
    def select_and_slice_target(feat, target_idx, slice_dim):
        """
        feat: [batch, num_padded_enemy/resource, enemy/resource_dim]
        target_idx: [batch, num_padded_unit]
        slice_dim: slice first k dim per feature
        """
        feat_dim = feat.size(2)
        target_feat = feat.gather(1, target_idx.unsqueeze(2).repeat(1, 1, feat_dim))
        sliced_feat = target_feat[:, :, :slice_dim]
        return sliced_feat

    def forward(self,
                num_real_unit,
                cmd_type,
                target_type,
                x,
                y,
                target_attack_idx,
                target_gather_idx,
                enemy_feat,
                resource_feat):
        """take cmd_type, produce cmd_repr

        num_real_unit:  [batch]
        cmd_type: [batch, num_padded_unit],
          0 <= cmd_type < num_cmd_type, padding cmd should have a valid unit_type
        target_type: [batch, num_padded_unit]
        x: [batch, num_padded_unit]
        y: [batch, num_padded_unit]
        target_idx: [batch, num_padded_unit]
        enemy_feat: [batch, num_padded_enemy, target_feat_dim]
        resource_feat: [batch, num_padded_resource, target_feat_dim]

        return: cmd_feat: [cmd_type_emb; target_type_emb; xy_feat; target_feat]
        """
        batch, pnum_unit = cmd_type.size()

        assert_eq(cmd_type.size(), target_type.size())
        assert_eq(x.dim(), 2)
        assert_eq(y.dim(), 2)

        cmd_type_emb = self.cmd_type_emb(cmd_type)
        target_type_emb = self.target_type_emb(target_type)
        xy_feat = self.xy_net(torch.stack([x, y], 2))

        enemy_feat = self.enemy_feat_net(enemy_feat)
        resource_feat = self.resource_feat_net(resource_feat)
        # select the targeted enemy's feature
        target_enemy_feat = self.select_and_slice_target(
            enemy_feat, target_attack_idx, self.attribute_dim)
        target_resource_feat = self.select_and_slice_target(
            resource_feat, target_gather_idx, self.attribute_dim)

        cmd_mask = torch.zeros(
            (batch, pnum_unit, self.num_cmd_type), device=cmd_type.device)
        cmd_mask.scatter_(2, cmd_type.unsqueeze(2), 1)
        # cmd_mask [batch, pnum_unit, num_cmd_type]

        outfeat = torch.zeros(
            (batch, pnum_unit, self.out_dim), device=cmd_type.device)

        outfeat[:, :, 0 : self.cmd_type_emb_end] = cmd_type_emb

        target_type_mask = (
            cmd_mask[:, :, gc.CmdTypes.BUILD_BUILDING.value]
            + cmd_mask[:, :, gc.CmdTypes.BUILD_UNIT.value]
        )
        outfeat[:, :, self.cmd_type_emb_end : self.target_type_emb_end] = (
            target_type_mask.unsqueeze(2) * target_type_emb
        )

        xy_feat_mask = (
            cmd_mask[:, :, gc.CmdTypes.BUILD_BUILDING.value]
            + cmd_mask[:, :, gc.CmdTypes.MOVE.value]
        )
        outfeat[:, :, self.target_type_emb_end : self.xy_feat_end] = (
            xy_feat_mask.unsqueeze(2) * xy_feat
        )

        enemy_feat_mask = cmd_mask[:, :, gc.CmdTypes.ATTACK.value]
        resource_feat_mask = cmd_mask[:, :, gc.CmdTypes.GATHER.value]
        outfeat[:, :, self.xy_feat_end : self.target_feat_end] = (
            enemy_feat_mask.unsqueeze(2) * target_enemy_feat
            + resource_feat_mask.unsqueeze(2) * target_resource_feat
        )
        return outfeat


def test_advanced_cmd_embedding():
    num_cmd_type = 6
    cmd_type_emb_dim = 2
    num_utype = 4
    utype_emb_dim = 2
    xy_feat_dim = 2
    target_feat_dim = 5
    cmd_emb = AdvancedCmdEmbedding(
        num_cmd_type,
        num_utype,
        1,
    )

    num_real_unit = torch.Tensor([[6]]).long()
    pnum_unit = 10
    cmd_type = torch.Tensor([[0, 1, 2, 3, 4, 5, 0, 0, 0, 0]]).long()
    target_type = torch.Tensor([[0, 1, 2, 3, 0, 0, 0, 0, 0, 0]]).long()
    x = torch.rand((1, pnum_unit))
    y = torch.rand((1, pnum_unit))
    target_attack_idx = torch.Tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).long()
    target_gather_idx = torch.Tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).long()
    enemy_feat = torch.rand((1, pnum_unit, target_feat_dim))
    resource_feat = torch.rand((1, pnum_unit, target_feat_dim))
    print('resource_feat')
    print(resource_feat)
    print('enemy_feat')
    print(enemy_feat)

    emb = cmd_emb.forward(
        num_real_unit,
        cmd_type,
        target_type,
        x,
        y,
        target_attack_idx,
        target_gather_idx,
        enemy_feat,
        resource_feat)
    print(emb)
    return emb


class MlpEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_hid, activate_out):
        """generic mlp encoder

        num_hid: # of hidden layers, excluding output layer
        activate_out: bool, whether apply ReLU on output
        """
        super().__init__()

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_hid = num_hid
        self.activate_out = activate_out

        self.net = self._create_net(
            in_dim, hid_dim, out_dim, num_hid, activate_out
        )

    @staticmethod
    def _create_net(in_dim, hid_dim, out_dim, num_hid, activate_out):
        if num_hid == -1:
            return None

        layers = []
        for i in range(num_hid):
            layers.append(weight_norm(nn.Linear(in_dim, hid_dim), dim=None))
            layers.append(nn.ReLU())
            in_dim = hid_dim

        layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
        if activate_out:
            layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.net is None:
            return x
        return self.net(x)
