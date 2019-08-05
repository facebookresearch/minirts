# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import pickle
import torch
import torch.nn as nn

from cmd_heads import *
from module import *
from conv_module import ConvFeatNet
from reducer import SumGlobReducer
from instruction_encoder import *

from common_utils import assert_eq, assert_lt
import common_utils.global_consts as gc


class UnitDropout(nn.Module):
    """apply same dropout mask for each unit"""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        """
        x: [batch, num_unit, unit_dim]
        """
        assert x.dim() == 3

        if not self.training:
            return x

        batch, num_unit, unit_dim = x.size()
        mask = torch.rand(batch, 1, unit_dim).to(x.device)
        mask = (mask >= self.dropout).float()
        x = x * mask.expand(batch, num_unit, unit_dim)
        x = x * 1 / (1 - self.dropout)
        return x


class ConvGlobEncoder(nn.Module):
    @staticmethod
    def get_arg_parser():
        """
        this function should only define args that are required sololy,
        and highly related to this class. other addtional params
        should still be passed via constructor
        i.e. private configs
        """
        parser = argparse.ArgumentParser()

        # network configs
        # unit specific features
        # type and hp embedding, shared by army, enemy, resource
        parser.add_argument('--num_hp_bins', type=int, default=11)
        parser.add_argument('--emb_field_dim', type=int, default=16)
        parser.add_argument('--prev_cmd_dim', type=int, default=64, help='')

        # conv net config
        parser.add_argument('--num_conv_layers',
                            type=int, default=3, help='# layers for conv')
        parser.add_argument('--num_post_layers',
                            type=int, default=1, help='# layers for post conv')
        parser.add_argument('--conv_in_dim',
                            type=int, default=42, help='hacked & hard coded')
        parser.add_argument('--conv_hid_dim', type=int, default=128)
        parser.add_argument('--army_out_dim', type=int, default=128)
        parser.add_argument('--other_out_dim', type=int, default=64)

        # encoder for money
        # parser.add_argument('--money_hid_dim',
        #                     type=int, default=64, help='hid size of money encoder')
        parser.add_argument('--money_hid_layer',
                            type=int, default=1, help='# hid layer of money encoder')

        # feature dropout
        parser.add_argument('--conv_dropout', type=float, default=0.0)

        # model selection
        parser.add_argument('--use_hist_inst', type=int, default=0)
        parser.add_argument('--pos_dim', type=int, default=32)
        parser.add_argument('--prev_cmd_rnn', type=int, default=0)

        return parser

    def __init__(self,
                 args,
                 num_unit_type,
                 num_cmd_type,
                 num_resource_bin,
                 inst_encoder):

        super().__init__()
        self.args = args

        self.unit_emb = UnitTypeHpEmbedding(
            num_unit_type,
            args.num_hp_bins,
            args.emb_field_dim,
        )
        if hasattr(args, 'prev_cmd_rnn') and args.prev_cmd_rnn:
            self.prev_cmd_emb = PrevCmdRnn(
                num_cmd_type,
                args.prev_cmd_dim,
            )
        else:
            self.prev_cmd_emb = PrevCmdEmbedding(
                num_cmd_type,
                args.prev_cmd_dim,
            )

        self.cmd_emb = AdvancedCmdEmbedding(
            num_cmd_type,
            num_unit_type,
            args.other_out_dim,
            args.emb_field_dim,
        )
        self.conv_encoder = ConvFeatNet(
            args.conv_in_dim,
            args.conv_hid_dim,
            args.army_out_dim,
            args.other_out_dim,
            args.num_conv_layers,
            args.num_post_layers,
        )

        self.enemy_extra_project = nn.utils.weight_norm(
            nn.Linear(self.unit_emb.out_dim, self.conv_encoder.other_out_dim), dim=None)
        self.resource_extra_project = nn.utils.weight_norm(
            nn.Linear(self.unit_emb.out_dim, self.conv_encoder.other_out_dim), dim=None)
        army_extra_dim = (
            self.unit_emb.out_dim + self.cmd_emb.out_dim + self.prev_cmd_emb.out_dim)
        self.army_extra_project = nn.utils.weight_norm(
            nn.Linear(army_extra_dim, self.conv_encoder.army_out_dim), dim=None)

        self.money_encoder = MlpEncoder(
            num_resource_bin,
            args.other_out_dim,
            args.other_out_dim,
            args.money_hid_layer,
            True
        )

        self.inst_encoder = inst_encoder
        self.inst_dim = self.inst_encoder.out_dim

        if hasattr(self.args, 'use_hist_inst') and self.args.use_hist_inst:
            self.hist_encoder = HistInstructionEncoder(
                inst_encoder,
                5,
                self.args.pos_dim,
                self.conv_encoder.army_out_dim
            )
            self.inst_dim = self.hist_encoder.out_dim

        self.dropout = nn.Dropout(args.conv_dropout)
        self.unit_dropout = UnitDropout(args.conv_dropout)

        self.army_dim = self.conv_encoder.army_out_dim
        self.enemy_dim = self.conv_encoder.other_out_dim
        self.resource_dim = self.conv_encoder.other_out_dim
        self.map_dim = self.conv_encoder.other_out_dim
        self.money_dim = self.conv_encoder.other_out_dim
        self.glob_dim = (self.army_dim
                         + self.enemy_dim
                         + self.resource_dim
                         + self.money_dim
                         + self.inst_dim)

        # self.inst_proj = nn.Sequential(
        #     weight_norm(nn.Linear(self.inst_dim, self.army_dim), dim=None),
        #     nn.ReLU(),
        # )

    @staticmethod
    def format_input(batch, prefix=''):
        # TODO: merge this with executor.format_input
        my_units = {
            'types': batch[prefix+'army_type'],
            'hps': batch[prefix+'army_hp'],
            'xs': batch[prefix+'army_x'],
            'ys': batch[prefix+'army_y'],
            'num_units': batch[prefix+'num_army'].squeeze(1),
        }
        enemy_units = {
            'types': batch[prefix+'enemy_type'],
            'hps': batch[prefix+'enemy_hp'],
            'xs': batch[prefix+'enemy_x'],
            'ys': batch[prefix+'enemy_y'],
            'num_units': batch[prefix+'num_enemy'].squeeze(1),
        }
        resource_units = {
            'types': batch[prefix+'resource_type'],
            'hps': batch[prefix+'resource_hp'],
            'xs': batch[prefix+'resource_x'],
            'ys': batch[prefix+'resource_y'],
            'num_units': batch[prefix+'num_resource'].squeeze(1),
        }
        current_cmds = {
            'cmd_type': batch[prefix+'current_cmd_type'],
            'target_type': batch[prefix+'current_cmd_unit_type'],
            'target_x': batch[prefix+'current_cmd_x'],
            'target_y': batch[prefix+'current_cmd_y'],
            'target_gather_idx': batch[prefix+'current_cmd_gather_idx'],
            'target_attack_idx': batch[prefix+'current_cmd_attack_idx'],
        }

        data = {
            # 'inst': inst,
            # 'inst_len': inst_len,
            'resource_bin': batch[prefix+'resource_bin'],
            'my_units': my_units,
            'enemy_units': enemy_units,
            'resource_units': resource_units,
            'prev_cmds': batch[prefix+'prev_cmd'],
            'current_cmds': current_cmds,
            'map': batch[prefix+'map'],
        }
        return data

    def _encode_army_extra(self, batch, enemy_feat, resource_feat):
        army_emb = self.unit_emb(batch['my_units']['types'], batch['my_units']['hps'])

        # get and append cmd related features
        current_cmd_emb = self.cmd_emb(
            batch['my_units']['num_units'],
            batch['current_cmds']['cmd_type'],
            batch['current_cmds']['target_type'],
            batch['current_cmds']['target_x'].float() / gc.MAP_X,
            batch['current_cmds']['target_y'].float() / gc.MAP_Y,
            batch['current_cmds']['target_attack_idx'],
            batch['current_cmds']['target_gather_idx'],
            enemy_feat,
            resource_feat)
        # current_cmd_feat = self.current_cmd_encoder(current_cmd_emb)
        # print(batch['prev_cmds'])
        prev_cmd_emb = self.prev_cmd_emb(batch['prev_cmds'], None)
        # prev_cmd_feat = self.prev_cmd_encoder(prev_cmd_emb)
        # print(army_emb.size())
        # print(current_cmd_emb.size())
        # print(prev_cmd_emb.size())
        extra = torch.cat([army_emb, current_cmd_emb, prev_cmd_emb], 2)
        extra = self.army_extra_project(extra)
        return extra

    def _encode_enemy_extra(self, batch):
        enemy_emb = self.unit_emb(batch['enemy_units']['types'], batch['enemy_units']['hps'])
        return self.enemy_extra_project(enemy_emb)

    def _encode_resource_extra(self, batch):
        resource_emb = self.unit_emb(
            batch['resource_units']['types'], batch['resource_units']['hps'])
        return self.resource_extra_project(resource_emb)

    def forward(self, batch, *, use_prev_inst=False):
        army_feat, enemy_feat, resource_feat, map_feat = self.conv_encoder(batch)

        enemy_extra = self._encode_enemy_extra(batch)
        enemy_feat = enemy_feat * enemy_extra

        resource_extra = self._encode_resource_extra(batch)
        resource_feat = resource_feat * resource_extra

        army_extra = self._encode_army_extra(batch, enemy_feat, resource_feat)
        army_feat = army_feat * army_extra

        money_feat = self.money_encoder(batch['resource_bin'])

        if hasattr(self.args, 'use_hist_inst') and self.args.use_hist_inst:
            assert not use_prev_inst
            inst_feat, sum_inst = self.hist_encoder(
                army_feat,
                batch['hist_inst'],
                batch['hist_inst_len'],
                batch['hist_inst_diff'])
            # print(inst_feat.size())
        else:
            if use_prev_inst:
                inst_feat = self.inst_encoder(batch['prev_inst'], batch['prev_inst_len'])
            else:
                # print('use normal inst')
                inst_feat = self.inst_encoder(batch['inst'], batch['inst_len'])
            sum_inst = inst_feat
            inst_feat = inst_feat.unsqueeze(1).repeat(1, army_feat.size(1), 1)
            # print('inst feat size:', inst_feat.size())

        # if self.dropout > 0:
        #     print(army_feat.size())
        if self.args.conv_dropout > 0:
            army_feat = self.unit_dropout(army_feat)
            enemy_feat = self.unit_dropout(enemy_feat)
            resource_feat = self.unit_dropout(resource_feat)
            money_feat = self.dropout(money_feat)
            # inst_feat = self.dropout(inst_feat)

        # glob_feat = self.glob_reducer(
        #     army_feat,
        #     batch['my_units']['num_units'],
        #     enemy_feat,
        #     batch['enemy_units']['num_units'],
        #     resource_feat,
        #     batch['resource_units']['num_units'],
        #     money_feat,
        #     inst_feat,
        #     # repeat_for_each_unit=not only_glob,
        # )

        # if only_glob:
        #     assert False
        #     return glob_feat

        # inst_proj = self.inst_proj(inst_feat)
        # if inst_proj.dim() == 2:
        #     # print(inst_proj.size())
        #     inst_proj = inst_proj.unsqueeze(1).repeat(1, army_feat.size(1), 1)
        # # print(inst_proj.size(), army_feat.size())
        # army_feat = army_feat * inst_proj
        # army_feat = torch.cat([army_feat, inst_proj], 2)
        # print(army_feat)

        batchsize, map_feat_dim, y, x = map_feat.size()
        map_feat = map_feat.view(batchsize, map_feat_dim, y * x)
        map_feat = map_feat.transpose(1, 2).contiguous()

        features = {
            'inst_feat': inst_feat,
            'army_feat': army_feat,
            'enemy_feat': enemy_feat,
            'resource_feat': resource_feat,
            'money_feat': money_feat,
            'map_feat': map_feat,
            'sum_inst': sum_inst,
            'sum_army': SumGlobReducer.sum(
                army_feat, batch['my_units']['num_units']),
            'sum_enemy': SumGlobReducer.sum(
                enemy_feat, batch['my_units']['num_units']),
            'sum_resource': SumGlobReducer.sum(
                resource_feat, batch['my_units']['num_units']),
        }
        return features
