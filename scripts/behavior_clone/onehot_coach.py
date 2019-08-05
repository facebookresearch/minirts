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
from instruction_selector import OneHotSelector


class ConvOneHotCoach(ConvRnnCoach):
    def __init__(self,
                 args,
                 max_raw_chars,
                 max_instruction_span,
                 num_resource_bin,
                 *,
                 num_unit_type=len(gc.UnitTypes),
                 num_cmd_type=len(gc.CmdTypes)):
        # 'rnn' is a hack, no use
        super().__init__(
            args, max_raw_chars, max_instruction_span, 'onehot', num_resource_bin)

        # overwrite inst_selector
        self.inst_selector = OneHotSelector(
            self.glob_feat_dim,
            self.args.inst_hid_dim,
            self.args.num_pos_inst,
        )

    @classmethod
    def load(cls, model_file):
        params = pickle.load(open(model_file + '.params', 'rb'))
        params.pop('coach_mode')
        model = cls(**params)
        model.load_state_dict(torch.load(model_file))
        return model

    def compute_loss(self, batch):
        """used for pre-training the model with dataset
        """
        glob_feat = self._forward(batch)

        cont = 1 - batch['is_base_frame']
        cont_loss = self.cont_cls.compute_loss(glob_feat, cont)
        lang_loss = self.inst_selector.compute_loss(glob_feat, batch['inst_idx'])

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
        return self.compute_loss(batch)

    def rl_forward(self, batch, mode=''):
        """forward function use by RL
        """
        # if not isinstance(batch, dict):
        #     batch = self.format_coach_input(batch)
        # In RL, the data is always in one-hot format, need to process
        batch = self._format_rl_language_input(batch)

        glob_feat = self._forward(batch)
        v = torch.tanh(self.value(glob_feat).squeeze())
        cont_prob = self.cont_cls.compute_prob(glob_feat)
        inst_prob = self.inst_selector.compute_prob(glob_feat)

        output = {
            'cont_pi': cont_prob,
            'inst_pi': inst_prob,
            'v': v
        }
        return output

    def _format_rl_language_input(self, batch):
        prev_inst, prev_inst_len = parse_batch_inst(
            self.inst_dict,
            batch['prev_inst_idx'].cpu().numpy(),
            batch['prev_inst_idx'].device)
        batch['prev_inst'] = prev_inst
        batch['prev_inst_len'] = prev_inst_len

        return batch
