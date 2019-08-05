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
from cmd_heads import create_real_unit_mask, masked_softmax



class SumGlobReducer(nn.Module):
    def __init__(self,
                 army_dim,
                 enemy_dim,
                 resource_dim,
                 money_dim,
                 instruction_dim):
        super().__init__()

        print('reducer dim:')
        print('\t army:', army_dim)
        print('\t enemy:', enemy_dim)
        print('\t resource:', resource_dim)
        print('\t money:', money_dim)
        print('\t instruction:', instruction_dim)

        # self.glob_dim = (
        #     army_dim + enemy_dim + resource_dim + instruction_dim
        # )
        # self.glob_dim = (
        #     # money_dim
        #     instruction_dim
        # )

    @staticmethod
    def sum(unit, num_unit):
        pnum_unit = unit.size(1)
        mask = create_real_unit_mask(num_unit, pnum_unit)
        masked_unit = mask.unsqueeze(2) * unit
        summed_unit = masked_unit.sum(1)
        return summed_unit

    def forward(self,
                army,
                num_army,
                enemy,
                num_enemy,
                resource,
                num_resource,
                money,
                instruction,
                *,
                repeat_for_each_unit=False):
        """reduce padded features of different categories and combine them

        army/enemy/resource: [batch, padded_num_army, army_out_dim]
        num_army/enemy/resource: [batch]

        return glob: [batch, padded_num_army, self.glob_dim]
        """
        pnum_army = army.size(1)
        summed_army = self._sum(army, num_army)
        summed_enemy = self._sum(enemy, num_enemy)
        summed_resource = self._sum(resource, num_resource)
        assert instruction.dim() == 2
        # money.zero_()
        # money = money + 1
        # print(money.size())
        if instruction.dim() == 2:
            glob = torch.cat([summed_army, summed_enemy, summed_resource, instruction], 1)
            # glob = torch.cat([money.detach()], 1)
            # glob = torch.cat([instruction], 1)
            # assert not repeat_for_each_unit
            if repeat_for_each_unit:
                glob = glob.unsqueeze(1).repeat(1, pnum_army, 1)

        elif instruction.dim() == 3:
            assert False

            glob = torch.cat(
                [summed_army, summed_enemy, summed_resource, money], 1)
            assert repeat_for_each_unit
            glob = glob.unsqueeze(1).repeat(1, pnum_army, 1)
            glob = torch.cat([glob, instruction], 2)
        else:
            assert False

        return glob
