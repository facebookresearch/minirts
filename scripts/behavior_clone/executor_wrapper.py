# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn

from common_utils import assert_eq, assert_lt
from utils import convert_to_raw_instruction
from instruction_encoder import is_word_based


def format_reply(batch, coach_reply, executor_reply):
    reply = coach_reply.copy()
    reply.update(executor_reply)
    reply['num_unit'] = batch['num_army']
    return reply


class ExecutorWrapper(nn.Module):
    def __init__(self, coach, executor, num_insts, max_raw_chars, cheat):
        super().__init__()
        self.coach = coach
        self.executor = executor
        if coach is not None:
            assert self.executor.inst_dict._idx2inst == self.coach.inst_dict._idx2inst

        self.num_insts = num_insts
        self.max_raw_chars = max_raw_chars
        self.cheat = cheat
        self.prev_inst = ''

    def _get_human_instruction(self, batch):
        assert_eq(batch['prev_inst'].size(0), 1)
        device = batch['prev_inst'].device

        inst = input('Please input your instruction\n')
        # inst = 'build peasant'

        inst_idx = torch.zeros((1,)).long().to(device)
        inst_idx[0] = self.executor.inst_dict.get_inst_idx(inst)
        inst_cont = torch.zeros((1,)).long().to(device)
        if len(inst) == 0:
            # inst = batch['prev_inst']
            inst = self.prev_inst
            inst_cont[0] = 1

        self.prev_inst = inst
        raw_inst = convert_to_raw_instruction(inst, self.max_raw_chars)
        inst, inst_len = self.executor.inst_dict.parse(inst, True)
        inst = torch.LongTensor(inst).unsqueeze(0).to(device)
        inst_len = torch.LongTensor([inst_len]).to(device)
        raw_inst = torch.LongTensor([raw_inst]).to(device)

        reply = {
            'inst': inst_idx.unsqueeze(1),
            'inst_pi': torch.ones(1, self.num_insts).to(device) / self.num_insts,
            'cont': inst_cont.unsqueeze(1),
            'cont_pi': torch.ones(1, 2).to(device) / 2,
            'raw_inst': raw_inst
        }

        return inst, inst_len, inst_cont, reply

    def forward(self, batch):
        if self.coach is not None:
            assert not self.coach.training
            if self.cheat:
                coach_input = self.coach.format_coach_input(batch, 'nofow_')
                # print(coach_input['enemy_units'])
            else:
                coach_input = self.coach.format_coach_input(batch)
            word_based = is_word_based(self.executor.args.inst_encoder_type)
            inst, inst_len, inst_cont, coach_reply = self.coach.sample(
                coach_input, word_based)
        else:
            inst, inst_len, inst_cont, coach_reply = self._get_human_instruction(batch)

        assert not self.executor.training
        executor_input = self.executor.format_executor_input(
            batch, inst, inst_len, inst_cont)
        executor_reply = self.executor.compute_prob(executor_input)

        reply = format_reply(batch, coach_reply, executor_reply)
        return reply
