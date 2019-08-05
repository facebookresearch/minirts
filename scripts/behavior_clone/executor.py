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
from instruction_encoder import *
from conv_glob_encoder import ConvGlobEncoder
from rnn_coach import parse_batch_inst

from common_utils import assert_eq, assert_lt
import common_utils.global_consts as gc


def parse_hist_inst(inst_dict, hist_inst, hist_inst_diff, inst, inst_len, inst_cont, word_based):
    device = hist_inst.device

    parsed_hist = []
    hist_len = []
    hist_diff = []
    bsize, num_inst = hist_inst.size()
    for bid in range(bsize):
        parsed_list = []
        diff_list = []
        l_list = []

        j = 0
        new_inst = (inst_cont[bid].item() == 0)
        if new_inst:
            j = 1
        while j < num_inst:
            idx = hist_inst[bid][j]
            if idx < 0:
                idx = inst_dict.pad_inst_idx
            if word_based:
                parsed, l = inst_dict.parse(inst_dict.get_inst(idx), True)
            else:
                parsed, l = idx, 0
            parsed_list.append(parsed)
            l_list.append(l)
            diff_list.append(hist_inst_diff[bid][j].item())
            j += 1

        if new_inst:
            if word_based:
                parsed_list.append(inst[bid].tolist())
            else:
                parsed_list.append(inst[bid].item())

            l_list.append(inst_len[bid].item())
            diff_list.append(0)

        # import pprint
        # pprint.pprint(parsed_list)
        # pprint.pprint(l_list)
        # pprint.pprint(diff_list)

        parsed_hist.append(parsed_list)
        hist_len.append(l_list)
        hist_diff.append(diff_list)

    parsed_hist = torch.LongTensor(parsed_hist).to(device)
    # print(parsed_hist)
    hist_len = torch.LongTensor(hist_len).to(device)
    hist_diff = torch.LongTensor(hist_diff).to(device)
    return parsed_hist, hist_len, hist_diff


def log_sum_exp(log1, log2):
    """compute the log(sum(exp(log1) + exp(log2)))
    """
    alpha = torch.max(log1, log2)
    # alpha = logs.max(dim, keepdim=True)[0] - _HALF_LOG_MAX #
    log1 = log1 - alpha
    log2 = log2 - alpha
    logs = alpha + torch.log(torch.exp(log1) + torch.exp(log2))
    return logs


def create_loss_masks(num_unit, cmd_type, num_cmd_type):
    """create loss mask for all type of cmds

    num_unit: [batch]
    cmd_type: [batch, padded_num_unit]
    num_cmd_type: int

    return
      real_unit_mask: [batch, padded_num_unit]
      cmd_mask: [batch, padded_num_unit, num_cmd_type]
    """
    batch, pnum_unit = cmd_type.size()
    cmd_mask = torch.zeros((batch, pnum_unit, num_cmd_type)).to(num_unit.device)
    cmd_mask.scatter_(2, cmd_type.unsqueeze(2), 1)
    # mask [batch, pnum_unit, num_cmd_type], need zero out padded unit
    real_unit_mask = create_real_unit_mask(num_unit, pnum_unit)
    # real_unit_maks [batch, pnum_unit]
    cmd_mask = cmd_mask * real_unit_mask.unsqueeze(2)
    return real_unit_mask.detach(), cmd_mask.detach()


def test_create_loss_masks():
    num_unit = torch.tensor([0, 1, 2])
    num_cmd_type = 3
    cmd_type = (torch.rand((3, 5)) * num_cmd_type).long()
    print(cmd_type)
    print('======')
    print(create_loss_masks(num_unit, cmd_type, num_cmd_type)[1])


class Executor(nn.Module):
    @staticmethod
    def get_arg_parser():
        # TODO: should add 'merge' function for parser instead?
        parser = ConvGlobEncoder.get_arg_parser()

        # inst encoder
        parser.add_argument('--rnn_word_emb_dim',
                            type=int, default=64, help='word emb dim for rnn encoder')
        parser.add_argument('--word_emb_dropout',
                            type=float, default=0, help='word emb dim for rnn encoder')
        parser.add_argument('--top_num_inst',
                            type=int, default=-1, help='num isnt for topn encoder')
        parser.add_argument('--inst_hid_dim',
                            type=int, default=64, help='hid size of inst encoder')
        parser.add_argument('--inst_encoder_type',
                            type=str, default='bow', help='type of instructions encoder')

        # dictionary related
        parser.add_argument('--inst_dict_path',
                            type=str, required=True, help='path to dictionary')
        parser.add_argument('--max_sentence_length', type=int, default=15)

        # # for the first term in loss
        # parser.add_argument('--cmd_type_hid_dim', type=int, default=128)
        # parser.add_argument('--cmd_type_dropout', type=float, default=0.0)
        # # for the second term in loss
        # parser.add_argument('--cmd_head_hid_dim', type=int, default=128)
        # parser.add_argument('--cmd_head_dropout', type=float, default=0.0)

        return parser

    def _load_inst_dict(self, inst_dict_path):
        print('loading cmd dict from: ', inst_dict_path)
        if inst_dict_path is None or inst_dict_path == '':
            return None

        inst_dict = pickle.load(open(inst_dict_path, 'rb'))
        inst_dict.set_max_sentence_length(self.args.max_sentence_length)
        return inst_dict

    def _create_inst_encoder(self):
        # instruction encoder
        # assert is_word_based(self.args.inst_encoder_type)
        dict_size = self.inst_dict.total_vocab_size
        padding_idx = self.inst_dict.pad_word_idx

        if self.args.inst_encoder_type == 'bow':
            encoder = MeanBOWInstructionEncoder(
                dict_size,
                self.args.inst_hid_dim,
                self.args.word_emb_dropout,
                padding_idx,
            )
        elif self.args.inst_encoder_type == 'lstm':
            encoder = LSTMInstructionEncoder(
                dict_size,
                self.args.rnn_word_emb_dim,
                self.args.word_emb_dropout,
                self.args.inst_hid_dim,
                padding_idx,
            )
        elif self.args.inst_encoder_type == 'gru':
            encoder = GRUInstructionEncoder(
                dict_size,
                self.args.rnn_word_emb_dim,
                self.args.word_emb_dropout,
                self.args.inst_hid_dim,
                padding_idx,
            )
        elif self.args.inst_encoder_type == 'zero':
            encoder = ZeroInstructionEncoder(
                self.args.inst_hid_dim,
            )
        elif self.args.inst_encoder_type == 'topn':
            assert self.args.top_num_inst > 0
            encoder = TopNInstructionEncoder(
                self.args.top_num_inst,
                self.args.word_emb_dropout,
                self.args.inst_hid_dim,
            )
        else:
            assert False, 'not supported yet'

        return encoder

    def __init__(self,
                 args,
                 num_resource_bin,
                 *,
                 num_unit_type=len(gc.UnitTypes),
                 num_cmd_type=len(gc.CmdTypes),
                 x_size=gc.MAP_X,
                 y_size=gc.MAP_Y):
        super().__init__()

        self.params = {
            'args': args,
            'num_resource_bin': num_resource_bin,
            'num_unit_type': num_unit_type,
            'num_cmd_type': num_cmd_type,
            'x_size': x_size,
            'y_size': y_size
        }
        self.args = args
        self.num_cmd_type = num_cmd_type

        self.inst_dict = self._load_inst_dict(self.args.inst_dict_path)
        self.inst_encoder = self._create_inst_encoder()

        self.conv_encoder = ConvGlobEncoder(
            args,
            num_unit_type,
            num_cmd_type,
            num_resource_bin,
            self.inst_encoder)

        # [TODO] for simpler config
        # self.args.cmd_type_hid_dim = self.conv_encoder.args.other_out_dim
        # self.args.cmd_type_dropout = 0.0
        # self.args.cmd_head_hid_dim = self.conv_encoder.args.other_out_dim
        # self.args.cmd_head_dropout = 0.0
        self.target_emb_dim = self.conv_encoder.args.other_out_dim

        # classifiers
        self.glob_cont_cls = GlobClsHead(
            self.conv_encoder.glob_dim,
            self.target_emb_dim,
            2,
        )

        self.cmd_type_cls = CmdTypeHead(
            self.conv_encoder.army_dim + self.conv_encoder.inst_dim,
            self.conv_encoder.glob_dim - self.conv_encoder.inst_dim,
            self.target_emb_dim,
            num_cmd_type,
        )
        self.gather_cls = DotGatherHead(
            self.conv_encoder.army_dim + self.conv_encoder.inst_dim,
            self.conv_encoder.resource_dim,
            0
        )
        self.attack_cls = DotAttackHead(
            self.conv_encoder.army_dim + self.conv_encoder.inst_dim,
            self.conv_encoder.enemy_dim,
            0
        )
        self.move_cls = DotMoveHead(
            self.conv_encoder.army_dim + self.conv_encoder.inst_dim,
            self.conv_encoder.map_dim,
            0,
            x_size,
            y_size
        )
        self.build_unit_cls = BuildUnitHead(
            self.conv_encoder.army_dim + self.conv_encoder.inst_dim,
            0,
            self.target_emb_dim,
            num_unit_type,
        )
        self.build_building_cls = DotBuildBuildingHead(
            self.conv_encoder.army_dim + self.conv_encoder.inst_dim,
            self.conv_encoder.map_dim,
            0,
            self.target_emb_dim,
            num_unit_type,
            x_size,
            y_size,
        )

    def format_executor_input(self, batch, inst, inst_len, inst_cont):
        """convert batch (in c++ format) to input used by executor

        inst: [batch, max_sentence_len], LongTensor, parsed
        inst_len: [batch]
        """
        assert_eq(inst.dim(), 2)
        assert_eq(inst_len.dim(), 1)
        # assert_eq(inst_cont.dim(), 1)

        my_units = {
            'types': batch['army_type'],
            'hps': batch['army_hp'],
            'xs': batch['army_x'],
            'ys': batch['army_y'],
            'num_units': batch['num_army'].squeeze(1),
        }
        enemy_units = {
            'types': batch['enemy_type'],
            'hps': batch['enemy_hp'],
            'xs': batch['enemy_x'],
            'ys': batch['enemy_y'],
            'num_units': batch['num_enemy'].squeeze(1),
        }
        resource_units = {
            'types': batch['resource_type'],
            'hps': batch['resource_hp'],
            'xs': batch['resource_x'],
            'ys': batch['resource_y'],
            'num_units': batch['num_resource'].squeeze(1),
        }
        current_cmds = {
            'cmd_type': batch['current_cmd_type'],
            'target_type': batch['current_cmd_unit_type'],
            'target_x': batch['current_cmd_x'],
            'target_y': batch['current_cmd_y'],
            'target_gather_idx': batch['current_cmd_gather_idx'],
            'target_attack_idx': batch['current_cmd_attack_idx'],
        }

        # print('prev cmd for executor: (num units: %d)' % batch['num_army'][0][0].item())
        # print(batch['prev_cmd'][0, :batch['num_army'][0][0].item()])
        prev_cmds = batch['prev_cmd']

        # print('inst:', inst)
        # print('hist inst')
        # print(batch['hist_inst'])
        # print(batch['hist_inst_diff'])

        # hist_inst = batch['hist_inst']
        # bsize, num_inst = hist_inst.size()
        # parsed_hist_inst, hist_inst_len = parse_batch_inst(
        #     self.inst_dict, hist_inst.view(-1), hist_inst.device)
        # parsed_hist_inst = parsed_hist_inst.view(bsize, num_inst, -1)
        # hist_inst_len = hist_inst_len.view(bsize, num_inst)
        # print(parsed_hist_inst)

        # print('@@@@@@@@@@, after correction')
        # import time
        # t = time.time()
        hist_inst, hist_inst_len, hist_inst_diff = parse_hist_inst(
            self.inst_dict,
            batch['hist_inst'],
            batch['hist_inst_diff'],
            inst,
            inst_len,
            inst_cont,
            is_word_based(self.args.inst_encoder_type))
        # print('time for executor format hist:', time.time() - t)
        # print('hist_inst:\n', hist_inst)
        # print('hist_inst_len:\n', hist_inst_len)
        # print('hist_inst_diff:\n', hist_inst_diff)
        # print('@@@@@@@@@@')

        data = {
            'inst': inst,
            'inst_len': inst_len,
            'hist_inst': hist_inst,
            'hist_inst_len': hist_inst_len,
            'hist_inst_diff': hist_inst_diff,
            'resource_bin': batch['resource_bin'],
            'my_units': my_units,
            'enemy_units': enemy_units,
            'resource_units': resource_units,
            'prev_cmds': prev_cmds,
            'current_cmds': current_cmds,
            'map': batch['map'],
        }
        return data

    def save(self, model_file):
        torch.save(self.state_dict(), model_file)
        pickle.dump(self.params, open(model_file + '.params', 'wb'))

    @classmethod
    def load(cls, model_file):
        params = pickle.load(open(model_file + '.params', 'rb'))
        print(params)
        model = cls(**params)
        model.load_state_dict(torch.load(model_file))
        return model

    def forward(self, batch):
        return self.compute_loss(batch, mean=False)

    def compute_loss(self, batch, *, mean=True):
        # army_feat, enemy_feat, resource_feat, glob_feat, map_feat = self.conv_encoder(batch)
        features = self.conv_encoder(batch)

        real_unit_mask, cmd_type_mask = create_loss_masks(
            batch['my_units']['num_units'],
            batch['target_cmds']['cmd_type'],
            self.num_cmd_type
        )

        # global continue classfier
        glob_feat = torch.cat([features['sum_army'],
                               features['sum_enemy'],
                               features['sum_resource'],
                               features['money_feat'],
                               features['sum_inst']], 1)
        glob_cont_loss = self.glob_cont_cls.compute_loss(
            glob_feat, batch['glob_cont']
        )

        army_inst = torch.cat([features['army_feat'], features['inst_feat']], 2)
        # action-arg classifiers
        gather_loss = self.gather_cls.compute_loss(
            army_inst,
            features['resource_feat'],
            None,
            batch['resource_units']['num_units'],
            batch['target_cmds']['target_gather_idx'],
            cmd_type_mask[:, :, gc.CmdTypes.GATHER.value]
        )
        attack_loss = self.attack_cls.compute_loss(
            army_inst,
            features['enemy_feat'],
            None,            # glob_feat,
            batch['enemy_units']['num_units'],
            batch['target_cmds']['target_attack_idx'],
            cmd_type_mask[:, :, gc.CmdTypes.ATTACK.value]
        )
        build_building_loss = self.build_building_cls.compute_loss(
            army_inst,
            features['map_feat'],
            None,            # glob_feat,
            batch['target_cmds']['target_type'],
            batch['target_cmds']['target_x'],
            batch['target_cmds']['target_y'],
            cmd_type_mask[:, :, gc.CmdTypes.BUILD_BUILDING.value]
        )
        build_unit_loss, nil_build_logp = self.build_unit_cls.compute_loss(
            army_inst,
            None,            # glob_feat,
            batch['target_cmds']['target_type'],
            cmd_type_mask[:, :, gc.CmdTypes.BUILD_UNIT.value],
            include_nil=True
        )
        move_loss = self.move_cls.compute_loss(
            army_inst,
            features['map_feat'],
            None,            # glob_feat,
            batch['target_cmds']['target_x'],
            batch['target_cmds']['target_y'],
            cmd_type_mask[:, :, gc.CmdTypes.MOVE.value]
        )

        # type loss
        ctype_context = torch.cat([features['sum_army'],
                                   features['sum_enemy'],
                                   features['sum_resource'],
                                   features['money_feat']], 1)
        cmd_type_logp = self.cmd_type_cls.compute_prob(army_inst, ctype_context, logp=True)
        # cmd_type_logp: [batch, num_unit, num_cmd_type]

        # extra continue
        # cmd_type_prob = self.cmd_type_cls.compute_prob(army_inst, ctype_context)
        # cont_type_prob = cmd_type_prob[:, :, gc.CmdTypes.CONT.value].clamp(max=1-1e-6)
        build_unit_type_logp = cmd_type_logp[:, :, gc.CmdTypes.BUILD_UNIT.value]
        extra_cont_logp = build_unit_type_logp + nil_build_logp
        # extra_cont_logp: [batch, num_unit]
        # print('extra cont logp size:', extra_cont_logp.size())
        # the following hack only works if CONT is the last one
        assert gc.CmdTypes.CONT.value == len(gc.CmdTypes) - 1
        assert extra_cont_logp.size() == cmd_type_logp[:, :, gc.CmdTypes.CONT.value].size()
        cont_logp = log_sum_exp(
            cmd_type_logp[:, :, gc.CmdTypes.CONT.value], extra_cont_logp)
        # cont_logp: [batch, num_unit]
        cmd_type_logp = torch.cat(
            [cmd_type_logp[:, :, :gc.CmdTypes.CONT.value], cont_logp.unsqueeze(2)], 2)
        # cmd_type_logp: [batch, num_unit, num_cmd_type]
        cmd_type_logp = cmd_type_logp.gather(
            2, batch['target_cmds']['cmd_type'].unsqueeze(2)).squeeze(2)
        # cmd_type_logp: [batch, num_unit]
        cmd_type_loss = -(cmd_type_logp * real_unit_mask).sum(1)

        # aggregate losses
        num_my_units_size = batch['my_units']['num_units'].size()
        assert_eq(glob_cont_loss.size(), num_my_units_size)
        assert_eq(cmd_type_loss.size(), num_my_units_size)
        assert_eq(move_loss.size(), num_my_units_size)
        assert_eq(attack_loss.size(), num_my_units_size)
        assert_eq(gather_loss.size(), num_my_units_size)
        assert_eq(build_unit_loss.size(), num_my_units_size)
        assert_eq(build_building_loss.size(), num_my_units_size)
        unit_loss = (cmd_type_loss
                     + move_loss
                     + attack_loss
                     + gather_loss
                     + build_unit_loss
                     + build_building_loss)

        unit_loss = (1 - batch['glob_cont'].float()) * unit_loss
        loss = glob_cont_loss + unit_loss

        all_loss = {
            'loss': loss.detach(),
            'unit_loss': unit_loss.detach(),
            'cmd_type_loss': cmd_type_loss.detach(),
            'move_loss': move_loss.detach(),
            'attack_loss': attack_loss.detach(),
            'gather_loss': gather_loss.detach(),
            'build_unit_loss': build_unit_loss.detach(),
            'build_building_loss': build_building_loss.detach(),
            'glob_cont_loss': glob_cont_loss.detach()
        }

        if mean:
            for k, v in all_losses.items():
                all_losses[k] = v.mean()
            loss = loss.mean()

        return loss, all_loss

    def compute_prob(self, batch):
        features = self.conv_encoder(batch)
        # army_feat, enemy_feat, resource_feat, glob_feat, map_feat = self.conv_encoder(batch)

        num_army = batch['my_units']['num_units']
        num_enemy = batch['enemy_units']['num_units']
        num_resource = batch['resource_units']['num_units']

        glob_feat = torch.cat([features['sum_army'],
                               features['sum_enemy'],
                               features['sum_resource'],
                               features['money_feat'],
                               features['sum_inst']], 1)
        # prob is omitted in variable names
        glob_cont = self.glob_cont_cls.compute_prob(glob_feat)
        # glob_cont = glob_cont[:, 1]

        army_inst = torch.cat([features['army_feat'], features['inst_feat']], 2)
        ctype_context = torch.cat([features['sum_army'],
                                   features['sum_enemy'],
                                   features['sum_resource'],
                                   features['money_feat']], 1)

        cmd_type = self.cmd_type_cls.compute_prob(army_inst, ctype_context)
        gather_idx = self.gather_cls.compute_prob(
            army_inst, features['resource_feat'], None, num_resource)
        attack_idx = self.attack_cls.compute_prob(
            army_inst, features['enemy_feat'], None, num_enemy)
        unit_type = self.build_unit_cls.compute_prob(army_inst, None)
        building_type, building_loc = self.build_building_cls.compute_prob(
            army_inst, features['map_feat'], None)
        move_loc = self.move_cls.compute_prob(army_inst, features['map_feat'], None)

        # for i in range(num_army[0]):
        #     print('Unit type:', batch['my_units']['types'][0][i].item())
        #     print('  cmd type')
        #     for j, t in enumerate(gc.CmdTypes):
        #         print('\t', t, '%.2f' % cmd_type[0][i][j])
        #     # print('cmd type prob:', cmd_type[0][i])
        #     print('  unit type')
        #     for j, t in enumerate(gc.UnitTypes):
        #         print('\t', t, '%.2f' % unit_type[0][i][j])
        #     print('  building type')
        #     for j, t in enumerate(gc.UnitTypes):
        #         print('\t', t, '%.2f' % building_type[0][i][j])
        #     print('---')
        #     # print('build unit prob:', unit_type[0][i])

        batchsize = num_army.size(0)
        probs = {
            'glob_cont_prob': glob_cont,
            'cmd_type_prob': cmd_type,
            'gather_idx_prob': gather_idx,
            'attack_idx_prob': attack_idx,
            'unit_type_prob': unit_type,
            'building_type_prob': building_type,
            'building_loc_prob': building_loc,
            'move_loc_prob': move_loc,
        }
        return probs
