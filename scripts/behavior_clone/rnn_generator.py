# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import pickle
import torch

from common_utils import assert_eq
import common_utils.global_consts as gc
from rnn_coach import ConvRnnCoach, parse_batch_inst
from instruction_generator import RnnLanguageGenerator


class RnnGenerator(ConvRnnCoach):
    def __init__(self,
                 args,
                 max_raw_chars,
                 max_instruction_span,
                 num_resource_bin,
                 *,
                 num_unit_type=len(gc.UnitTypes),
                 num_cmd_type=len(gc.CmdTypes)):
        super().__init__(
            args, max_raw_chars, max_instruction_span, 'rnn_gen', num_resource_bin)

        # overwrite inst_selector
        self.inst_selector = RnnLanguageGenerator(
            self.prev_inst_encoder.emb,
            self.args.word_emb_dim,
            self.glob_feat_dim,
            self.args.inst_hid_dim,
            self.inst_dict.total_vocab_size,
            self.inst_dict,
        )

    @classmethod
    def load(cls, model_file):
        params = pickle.load(open(model_file + '.params', 'rb'))
        params.pop('coach_mode')
        print(params)
        model = cls(**params)
        model.load_state_dict(torch.load(model_file))
        return model

    def compute_loss(self, batch):
        """used for pre-training the model with dataset
        """
        batch = self._format_language_input(batch)
        glob_feat = self._forward(batch)

        cont = 1 - batch['is_base_frame']
        cont_loss = self.cont_cls.compute_loss(glob_feat, cont)
        lang_loss = self.inst_selector.compute_loss(
            batch['inst_input'],
            batch['inst'],
            glob_feat,
        )

        assert_eq(cont_loss.size(), lang_loss.size())
        lang_loss = (1 - cont.float()) * lang_loss
        loss = cont_loss + lang_loss
        loss = loss.mean()
        all_loss = {
            'loss': loss,
            'cont_loss': cont_loss.mean(),
            'lang_loss': lang_loss.mean()
        }
        return loss, all_loss

    def compute_eval_loss(self, batch):
        batch = self._format_language_input_with_candidate(batch)
        glob_feat = self._forward(batch)

        cont = 1 - batch['is_base_frame']
        cont_loss = self.cont_cls.compute_loss(glob_feat, cont)

        lang_logp = self.inst_selector.compute_prob(
            batch['inst_input'],
            batch['inst'],
            glob_feat,
            log=True
        )
        lang_loss = -lang_logp.gather(1, batch['inst_idx'].unsqueeze(1)).squeeze(1)

        assert_eq(cont_loss.size(), lang_loss.size())
        lang_loss = (1 - cont.float()) * lang_loss
        loss = cont_loss + lang_loss
        loss = loss.mean()
        all_loss = {
            'loss': loss,
            'cont_loss': cont_loss.mean(),
            'lang_loss': lang_loss.mean()
        }
        return loss, all_loss

    def _format_language_input(self, batch):
        """convert prev_inst and inst from one hot to rnn format,
        add inst_input for RNN
        """
        inst = batch['inst']

        start = torch.zeros(inst.size(0), 1) + self.inst_dict.start_word_idx
        start = start.long().to(inst.device)
        inst_input = torch.cat([start, inst[:, :-1]], 1)
        batch['inst_input'] = inst_input

        return batch

    def _format_language_input_with_candidate(self, batch):
        """convert prev_inst and inst from one hot to rnn format,
        add inst_input for RNN
        """
        inst, _ = self._get_pos_candidate_inst(batch['inst'].device)
        batch['inst'] = inst

        start = torch.zeros(inst.size(0), 1) + self.inst_dict.start_word_idx
        start = start.long().to(inst.device)
        inst_input = torch.cat([start, inst[:, :-1]], 1)
        batch['inst_input'] = inst_input

        return batch

    def _get_pos_candidate_inst(self, device):
        if (self.pos_candidate_inst is not None
            and self.pos_candidate_inst[0].device == device):
            inst, inst_len = self.pos_candidate_inst
        else:
            inst, inst_len = parse_batch_inst(
                self.inst_dict, range(self.args.num_pos_inst), device)
            self.pos_candidate_inst = (inst, inst_len)

        return inst, inst_len

    def rl_forward(self, batch, mode):
        """forward function use by RL
        """
        batch = self._format_rl_language_input(batch)
        glob_feat = self._forward(batch)
        v = self.value(glob_feat).squeeze()
        cont_prob = self.cont_cls.compute_prob(glob_feat)
        inst_prob = self.inst_selector.compute_prob(
            batch['cand_inst_input'], batch['cand_inst'], glob_feat)

        output = {
            'cont_pi': cont_prob,
            'inst_pi': inst_prob,
            'v': v
        }
        return output

    def _format_rl_language_input(self, batch):
        prev_inst, prev_inst_len = self._parse_batch_inst(
            batch['prev_inst_idx'].cpu().numpy(), batch['prev_inst_idx'].device)
        batch['prev_inst'] = prev_inst
        batch['prev_inst_len'] = prev_inst_len

        inst, inst_len = self._get_pos_candidate_inst(prev_inst.device)
        batch['cand_inst'] = inst

        start = torch.zeros(inst.size(0), 1) + self.inst_dict.start_word_idx
        start = start.long().to(inst.device)
        inst_input = torch.cat([start, inst[:, :-1]], 1)
        batch['cand_inst_input'] = inst_input

        return batch

    def _parse_batch_inst(self, indices, device):
        inst = []
        inst_len = []
        for idx in indices:
            parsed, l = self.inst_dict.parse(self.inst_dict.get_inst(idx), True)
            inst.append(parsed)
            inst_len.append(l)

        inst = torch.LongTensor(inst).to(device)
        inst_len = torch.LongTensor(inst_len).to(device)
        return inst, inst_len
