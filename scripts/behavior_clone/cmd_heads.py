# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import weight_norm

from common_utils import assert_eq


def logit2logp(logit, index):
    """return log_softmax(logit)[index] along dim=2

    logit: [batch, padded_num_unit, dim]
    index: [batch, padded_num_unit]

    return logp: [batch, padded_num_unit]
    """
    assert_eq(logit.size()[:2], index.size())
    assert_eq(logit.dim(), 3)

    logp = nn.functional.log_softmax(logit, 2)
    logp = logp.gather(2, index.unsqueeze(2)).squeeze(2)
    return logp


def sample_categorical(logit):
    """sample categorical distribution given pre-softmax logit alone dim=-1

    logit: [batch, padded_num_unit, dim]
    return: sample: [batch, padded_num_unit]
    """
    prob = logit.softmax(logit.dim() - 1)
    return torch.distributions.Categorical(probs=prob).sample()


def create_real_unit_mask(num_unit, padded_num):
    """create mask for real/padded unit

    num_unit: [batch]
    padded_num: int

    return mask: [batch, padded_num], mask[i,j] = 1 iff unit_{i,j} is real unit
    """
    batch = num_unit.size(0)
    mask = torch.arange(0, padded_num, device=num_unit.device)
    mask = mask.unsqueeze(0).repeat(batch, 1)
    # mask [batch, padded_num]
    mask = mask < num_unit.unsqueeze(1)
    return mask.float().detach()

def masked_softmax(logit, mask, dim):
    """masked softmax

    (assume dim = 3)
    logit: [batch, pnum_unit, pnum_enemy]
    mask: [batch, pnum_unit, pnum_enemy]
      if mask[batch, pnum_unit] == 0 for all (i.e. no enemy),
      will return uniform probability for that entry

    prob: [batch, pnum_unit, pnum_enemy]
    """
    assert_eq(logit.size(), mask.size())

    logit = logit - (1 - mask) * 1e9
    logit_max = logit.max(dim, keepdim=True)[0]#.detach()
    exp = (logit - logit_max).exp()
    denom = exp.sum(dim, keepdim=True)
    prob = exp / exp.sum(dim, keepdim=True)
    return prob


class BuildUnitHead(nn.Module):
    def __init__(self, ufeat_dim, globfeat_dim, hid_dim, out_dim):
        super().__init__()
        self.ufeat_dim = ufeat_dim
        self.globfeat_dim = globfeat_dim
        self.in_dim = ufeat_dim + globfeat_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim

        self.net = nn.Sequential(
            weight_norm(nn.Linear(self.in_dim, self.hid_dim), dim=None),
            nn.ReLU(),
            nn.Linear(self.hid_dim, self.out_dim, bias=False)
        )

    def forward(self, ufeat, globfeat):
        if globfeat is None:
            assert self.globfeat_dim == 0
            infeat = ufeat
        else:
            assert globfeat.dim() == 2
            globfeat = globfeat.unsqueeze(1).repeat(1, ufeat.size(1), 1)
            infeat = torch.cat([ufeat, globfeat], 2)

        logit = self.net(infeat)
        return logit

    def compute_loss(self, ufeat, globfeat, target_type, mask, *, include_nil=False):
        """loss, averaged by sum(mask)

        ufeat: [batch, padded_num_unit, ufeat_dim]
        globfeat: [batch, padded_num_unit, globfeat_dim]
        target_type: [batch, padded_num_unit]
          target_type[i, j] is real unit_type iff unit_{i,j} is build unit
        mask: [batch, padded_num_unit]
          mask[i, j] = 1 iff the unit is true unit and its cmd_type == BUILD_UNIT
        """
        batch, pnum_unit, _  = ufeat.size()
        assert_eq(target_type.size(), (batch, pnum_unit))
        assert_eq(mask.size(), (batch, pnum_unit))

        logit = self.forward(ufeat, globfeat)
        # logit [batch, pnum_unit, num_unit_type]
        logp = logit2logp(logit, target_type)
        # logp [batch, pnum_unit]
        loss = -(logp * mask)
        # if sum_loss:
        loss = loss.sum(1)
        if not include_nil:
            return loss

        nil_type = torch.zeros_like(target_type)
        nil_logp = logit2logp(logit, nil_type)
        return loss, nil_logp

    def compute_prob(self, ufeat, globfeat, *, logp=False):
        logit = self.forward(ufeat, globfeat)
        # logit [batch, pnum_unit, num_unit_type]
        if logp:
            logp = nn.functional.log_softmax(logit, 2)
            return logp
        else:
            prob = nn.functional.softmax(logit, 2)
            return prob

    def sample(self, ufeat, globfeat, greedy):
        logit = self.forward(ufeat, globfeat)
        if greedy:
            target_type = logit.max(2)[1]
        else:
            target_type = sample_categorical(logit)
        return target_type


CmdTypeHead = BuildUnitHead


class DotBuildBuildingHead(nn.Module):
    def __init__(self,
                 ufeat_dim,
                 mapfeat_dim,
                 globfeat_dim,
                 hid_dim,
                 out_dim,
                 x_size,
                 y_size):
        super().__init__()

        self.type_net = BuildUnitHead(
            ufeat_dim, globfeat_dim, hid_dim, out_dim)
        self.xy_net = DotMoveHead(
            ufeat_dim, mapfeat_dim, globfeat_dim, x_size, y_size)

    def compute_loss(self, ufeat, mapfeat, globfeat, target_type, x, y, mask):
        """loss, averaged by sum(mask)

        ufeat: [batch, padded_num_unit, ufeat_dim]
        globfeat: [batch, padded_num_unit, globfeat_dim]
        target_type: [batch, padded_num_unit]
          target_type[i, j] is real unit_type iff unit_{i,j} is build building
        x: [batch, padded_num_unit]
          same as above
        y: [batch, padded_num_unit]
          same as above
        mask: [batch, padded_num_unit]
          mask[i, j] = 1 iff the unit is true unit and its cmd_type == BUILD_BUILDING
        """
        type_loss = self.type_net.compute_loss(ufeat, globfeat, target_type, mask)
        xy_loss = self.xy_net.compute_loss(ufeat, mapfeat, globfeat, x, y, mask)
        loss = type_loss + xy_loss
        return loss

    def compute_prob(self, ufeat, mapfeat, globfeat):
        type_prob = self.type_net.compute_prob(ufeat, globfeat)
        loc_prob = self.xy_net.compute_prob(ufeat, mapfeat, globfeat)
        return type_prob, loc_prob

    def sample(self, ufeat, mapfeat, globfeat, greedy):
        target_type = self.type_net.sample(ufeat, globfeat, greedy)
        x, y = self.xy_net.sample(ufeat, mapfeat, globfeat, greedy)
        return target_type, x, y


class DotMoveHead(nn.Module):
    def __init__(self, ufeat_dim, mapfeat_dim, globfeat_dim, x_size, y_size):
        super().__init__()

        self.ufeat_dim = ufeat_dim
        self.mapfeat_dim = mapfeat_dim
        self.globfeat_dim = globfeat_dim
        self.x_size = x_size
        self.y_size = y_size
        self.in_dim = ufeat_dim + globfeat_dim
        self.norm = float(np.sqrt(self.mapfeat_dim))

        self.net = nn.Sequential(
            weight_norm(nn.Linear(self.in_dim, self.mapfeat_dim), dim=None),
            nn.ReLU(),
        )

    def forward(self, ufeat, mapfeat, globfeat):
        """
        ufeat: [batch, pnum_unit, ufeat_dim]
        mapfeat: [batch, x*y, mapfeat_dim]
        globfeat: [batch, pnum_unit, globfeat_dim]

        return logit: [batch, pnum_unit, mapfeat_dim]
        """
        pnum_unit = ufeat.size(1)
        map_dim = mapfeat.size(1)

        assert globfeat is None and self.globfeat_dim == 0
        infeat = ufeat#, globfeat], 2)
        proj = self.net(infeat)
        proj = proj.unsqueeze(2).repeat(1, 1, map_dim, 1)
        mapfeat = mapfeat.unsqueeze(1).repeat(1, pnum_unit, 1, 1)
        logit = (proj * mapfeat).sum(3) / self.norm
        return logit

    def compute_loss(self, ufeat, mapfeat, globfeat, x, y, mask):
        """loss
        """
        # y = y // 2
        # x = x // 2
        loc = y * self.x_size + x
        logit = self.forward(ufeat, mapfeat, globfeat)
        logp = logit2logp(logit, loc)
        loss = -(logp * mask).sum(1)
        return loss

    def compute_prob(self, ufeat, mapfeat, globfeat):
        logit = self.forward(ufeat, mapfeat, globfeat)
        # logit: [batch, pnum_unit, map_y * map_x]
        prob = nn.functional.softmax(logit, 2)
        return prob

    def sample(self, ufeat, mapfeat, globfeat, greedy):
        logit = self.forward(ufeat, mapfeat, globfeat)
        if greedy:
            loc = logit.max(2)[1]
        else:
            loc = sample_categorical(logit)

        y = loc // self.x_size
        x = loc % self.x_size
        # assert(y < self.y_size)
        # assert(x < self.x_size)
        return x, y


class DotAttackHead(nn.Module):
    def __init__(self, ufeat_dim, efeat_dim, globfeat_dim):
        super().__init__()
        self.ufeat_dim = ufeat_dim
        self.efeat_dim = efeat_dim
        self.globfeat_dim = globfeat_dim
        self.in_dim = ufeat_dim + globfeat_dim
        self.norm = float(np.sqrt(self.efeat_dim))

        self.net = nn.Sequential(
            weight_norm(nn.Linear(self.in_dim, self.efeat_dim), dim=None),
            nn.ReLU(),
        )

    def forward(self, ufeat, efeat, globfeat, num_enemy):
        """return masked prob that each real enemy is the target

        return: prob: [batch, pnum_unit, pnum_enemy]
        """
        batch, pnum_unit, _ = ufeat.size()
        pnum_enemy = efeat.size(1)
        assert_eq(num_enemy.size(), (batch,))

        assert globfeat is None and self.globfeat_dim == 0
        infeat = ufeat
        # infeat [batch, pnum_unit, in_dim]
        proj = self.net(infeat)
        proj = proj.unsqueeze(2).repeat(1, 1, pnum_enemy, 1)
        # proj [batch, pnum_unit, pnum_enemy, efeat_dim]
        efeat = efeat.unsqueeze(1).repeat(1, pnum_unit, 1, 1)
        # efeat [batch, pnum_unit, pnum_enemy, efeat_dim
        logit = (proj * efeat).sum(3) / self.norm
        # logit [batch, pnum_unit, pnum_enemy]
        enemy_mask = create_real_unit_mask(num_enemy, pnum_enemy)
        # enemy_mask [batch, pnum_enemy]
        enemy_mask = enemy_mask.unsqueeze(1).repeat(1, pnum_unit, 1)
        # if torch.isinf(logit).any() or torch.isnan(logit).any():
        #     import pdb
        #     pdb.set_trace()

        prob = masked_softmax(logit, enemy_mask, 2)
        # prob [batch, pnum_unit, pnum_enemy]
        return prob

    def compute_loss(self, ufeat, efeat, globfeat, num_enemy, target_idx, mask):
        """loss, averaged by sum(mask)

        ufeat: [batch, padded_num_unit, ufeat_dim]
        efeat: [batch, padded_num_enemy, efeat_dim]
        globfeat: [batch, padded_num_unit, globfeat_dim]
        num_enemy: [batch]
        target_idx: [batch, padded_num_unit]
          target_idx[i, j] is real target idx iff unit_{i,j} is attacking
        mask: [batch, padded_num_unit]
          mask[i] = 1 iff unit i is true unit and its cmd_type == ATTACK
        """
        if target_idx.min() < 0 or (target_idx.max(1)[0] > num_enemy).any():
            import pdb
            pdb.set_trace()

        batch, pnum_unit, _ = ufeat.size()
        assert_eq(target_idx.size(), (batch, pnum_unit))
        assert_eq(mask.size(), (batch, pnum_unit))

        prob = self.forward(ufeat, efeat, globfeat, num_enemy)
        # prob [batch, pnum_unit, pnum_enemy]
        prob = prob.gather(2, target_idx.unsqueeze(2)).squeeze(2)
        # prob [batch, pnum_unit]
        logp = (prob + 1e-6).log()

        loss = -(logp * mask).sum(1)
        return loss

    def compute_prob(self, ufeat, efeat, globfeat, num_enemy):
        return self.forward(ufeat, efeat, globfeat, num_enemy)

    def sample(self, ufeat, efeat, globfeat, num_enemy, greedy):
        prob = self.forward(ufeat, efeat, globfeat, num_enemy)
        if greedy:
            target_idx = prob.max(2)[1]
        else:
            target_idx = self._sample_softmax_prob(prob)
        return target_idx

    @staticmethod
    def _sample_softmax_prob(prob):
        """sample from probability produced by masked softmax

        prob: [batch, pnum_unit, pnum_enemy]
        num_enemy: [batch]

        """
        assert prob.min() >= 0, 'input should be prob'
        sample = torch.distributions.Categorical(probs=prob).sample()
        return sample


DotGatherHead = DotAttackHead


class GlobClsHead(nn.Module):
    def __init__(self, globfeat_dim, hid_dim, out_dim):
        super().__init__()
        self.globfeat_dim = globfeat_dim
        self.in_dim = globfeat_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim

        self.net = nn.Sequential(
            weight_norm(nn.Linear(self.in_dim, self.hid_dim), dim=None),
            nn.ReLU(),
            weight_norm(nn.Linear(self.hid_dim, self.out_dim), dim=None)
        )

    def forward(self, globfeat):
        logit = self.net(globfeat)
        return logit

    def compute_loss(self, globfeat, target):
        """loss, averaged by sum(mask)

        globfeat: [batch, globfeat_dim]
        target: [batch]
          target[i] = 1 iff the frame is glob_cont
        """
        logit = self.forward(globfeat)
        # logit [batch, 2]
        logp = nn.functional.log_softmax(logit, 1)
        logp = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        loss = -logp
        return loss

    def compute_prob(self, globfeat, *, temp=1, log=False):
        logit = self.forward(globfeat)
        logit = logit / temp
        # logit [batch, 2]
        if log:
            return nn.functional.log_softmax(logit, 1)
        else:
            return nn.functional.softmax(logit, 1)

    # def sample(self, globfeat, greedy):
    #     logit = self.forward(globfeat)
    #     # print('>>>glob cont prob:', logit.exp() / logit.exp().sum())
    #     if greedy:
    #         target_type = logit.max(1)[1]
    #     else:
    #         target_type = sample_categorical(logit)
    #     return target_type
