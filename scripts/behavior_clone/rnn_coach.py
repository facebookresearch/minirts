# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import pickle
import numpy as np
import torch
import torch.nn as nn

from common_utils import assert_eq
import common_utils.global_consts as gc

from cmd_heads import GlobClsHead
from module import MlpEncoder
from instruction_selector import RnnSelector
from utils import convert_to_raw_instruction

from coach_dataset import CoachDataset
from instruction_encoder import LSTMInstructionEncoder
from instruction_encoder import MeanBOWInstructionEncoder
from conv_glob_encoder import ConvGlobEncoder

from cont_softmax_sampler import ContSoftmaxSampler


def parse_batch_inst(inst_dict, indices, device):
    inst = []
    inst_len = []
    for idx in indices:
        if idx < 0:
            idx = inst_dict.pad_inst_idx
        parsed, l = inst_dict.parse(inst_dict.get_inst(idx), True)
        inst.append(parsed)
        inst_len.append(l)

    inst = torch.LongTensor(inst).to(device)
    inst_len = torch.LongTensor(inst_len).to(device)
    return inst, inst_len


class ConvRnnCoach(nn.Module):
    @staticmethod
    def get_arg_parser():
        parser = ConvGlobEncoder.get_arg_parser()

        # data related
        parser.add_argument('--inst_dict_path',
                            type=str, required=True, help='path to dictionary')
        parser.add_argument('--max_sentence_length', type=int, default=15)
        parser.add_argument('--num_pos_inst', type=int, default=50)
        parser.add_argument('--num_neg_inst', type=int, default=50)

        # prev_inst encoder
        parser.add_argument('--word_emb_dim', type=int, default=32)
        parser.add_argument('--word_emb_dropout', type=float, default=0.0)
        parser.add_argument('--inst_hid_dim', type=int, default=128)

        # count feat encoder
        parser.add_argument('--num_count_channels',
                            type=int, default=CoachDataset.get_num_count_channels())
        parser.add_argument('--count_hid_dim', type=int, default=128)
        parser.add_argument('--count_hid_layers', type=int, default=2)

        parser.add_argument('--glob_dropout', type=float, default=0.0)

        return parser

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

    def load_inst_dict(self, inst_dict_path):
        print('loading cmd dict from: ', inst_dict_path)
        if inst_dict_path is None or inst_dict_path == '':
            return None

        inst_dict = pickle.load(open(inst_dict_path, 'rb'))
        inst_dict.set_max_sentence_length(self.args.max_sentence_length)
        return inst_dict

    def __init__(self,
                 args,
                 max_raw_chars,
                 max_instruction_span,
                 coach_mode,
                 num_resource_bin,
                 *,
                 num_unit_type=len(gc.UnitTypes),
                 num_cmd_type=len(gc.CmdTypes)):
        super().__init__()

        self.params = {
            'args': args,
            'max_raw_chars': max_raw_chars,
            'max_instruction_span':  max_instruction_span,
            'coach_mode': coach_mode,
            'num_resource_bin': num_resource_bin,
            'num_unit_type': num_unit_type,
            'num_cmd_type': num_cmd_type,
        }

        self.args = args
        self.max_raw_chars = max_raw_chars
        self.max_instruction_span = max_instruction_span
        self.coach_mode = coach_mode

        self.pos_candidate_inst = None
        self.neg_candidate_inst = None

        self.args.inst_dict_path = self.args.inst_dict_path.replace(
            'scratch/rts_data', 'rts-replays')
        self.inst_dict = self.load_inst_dict(self.args.inst_dict_path)

        self.prev_inst_encoder = LSTMInstructionEncoder(
            self.inst_dict.total_vocab_size,
            self.args.word_emb_dim,
            self.args.word_emb_dropout,
            self.args.inst_hid_dim,
            self.inst_dict.pad_word_idx,
        )

        self.glob_encoder = ConvGlobEncoder(
            args,
            num_unit_type,
            num_cmd_type,
            num_resource_bin,
            self.prev_inst_encoder)

        # count encoders
        self.count_encoder = MlpEncoder(
            self.args.num_count_channels * 2,
            self.args.count_hid_dim,
            self.args.count_hid_dim,
            self.args.count_hid_layers - 1,
            activate_out=True
        )
        self.cons_count_encoder = MlpEncoder(
            num_unit_type,
            self.args.count_hid_dim // 2,
            self.args.count_hid_dim // 2,
            self.args.count_hid_layers - 1,
            activate_out=True
        )
        self.moving_avg_encoder = MlpEncoder(
            num_unit_type,
            self.args.count_hid_dim // 2,
            self.args.count_hid_dim // 2,
            self.args.count_hid_layers - 1,
            activate_out=True
        )
        self.frame_passed_encoder = nn.Embedding(
            max_instruction_span + 2,
            self.args.count_hid_dim // 2,
        )

        if self.args.glob_dropout > 0:
            self.glob_dropout = nn.Dropout(self.args.glob_dropout)

        self.glob_feat_dim = int(
            2.5 * self.args.count_hid_dim
            + self.glob_encoder.glob_dim
        )
        self.cont_cls = GlobClsHead(
            self.glob_feat_dim,
            self.args.inst_hid_dim,  # for reducing hyper-parameter
            2
        )

        if self.coach_mode == 'rnn':
            encoder = self.prev_inst_encoder
        elif self.coach_mode == 'bow':
            encoder = MeanBOWInstructionEncoder(
                self.inst_dict.total_vocab_size,
                self.args.inst_hid_dim,
                self.args.word_emb_dropout,
                self.inst_dict.pad_word_idx)
        elif self.coach_mode == 'onehot' or self.coach_mode == 'rnn_gen':
            pass
        else:
            assert False, 'unknown coach mode: %s' % self.coach_mode

        if self.coach_mode == 'rnn' or self.coach_mode == 'bow':
            self.inst_selector = RnnSelector(encoder, self.glob_feat_dim)
        else:
            self.inst_selector = None

        self.value = nn.utils.weight_norm(
            nn.Linear(self.glob_feat_dim, 1), dim=None
        )
        self.sampler = ContSoftmaxSampler(
            'cont', 'cont_pi', 'inst', 'inst_pi')

    @property
    def num_instructions(self):
        return self.args.num_pos_inst

    def _forward(self, batch):
        """shared forward function to compute glob feature
        """
        count_input = torch.cat(
            [batch['count'], batch['base_count'] - batch['count']], 1)
        count_feat = self.count_encoder(count_input)
        cons_count_feat = self.cons_count_encoder(batch['cons_count'])
        moving_avg_feat = self.moving_avg_encoder(batch['moving_enemy_count'])
        frame_passed_feat = self.frame_passed_encoder(batch['frame_passed'])
        features = self.glob_encoder(batch, use_prev_inst=True)

        glob = torch.cat([
            features['sum_inst'],
            features['sum_army'],
            features['sum_enemy'],
            features['sum_resource'],
            features['money_feat']], dim=1)

        glob_feat = torch.cat([
            glob,
            count_feat,
            cons_count_feat,
            moving_avg_feat,
            frame_passed_feat,
        ], dim=1)

        if self.args.glob_dropout > 0:
            glob_feat = self.glob_dropout(glob_feat)

        return glob_feat

    def compute_loss(self, batch):
        """used for pre-training the model with dataset
        """
        batch = self._format_supervised_language_input(batch)
        glob_feat = self._forward(batch)

        cont = 1 - batch['is_base_frame']
        cont_loss = self.cont_cls.compute_loss(glob_feat, cont)
        lang_loss = self.inst_selector.compute_loss(
            batch['pos_cand_inst'],
            batch['pos_cand_inst_len'],
            batch['neg_cand_inst'],
            batch['neg_cand_inst_len'],
            batch['inst'],
            batch['inst_len'],
            glob_feat,
            batch['inst_idx'])

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
        batch = self._format_supervised_language_input(batch)
        glob_feat = self._forward(batch)

        cont = 1 - batch['is_base_frame']
        cont_loss = self.cont_cls.compute_loss(glob_feat, cont)
        lang_loss = self.inst_selector.eval_loss(
            batch['pos_cand_inst'],
            batch['pos_cand_inst_len'],
            glob_feat,
            batch['inst_idx']
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

    def _format_supervised_language_input(self, batch):
        device = batch['prev_inst'].device
        pos_inst, pos_inst_len = self._get_pos_candidate_inst(device)
        neg_inst, neg_inst_len = self._get_neg_candidate_inst(device, batch['inst_idx'])
        batch['pos_cand_inst'] = pos_inst
        batch['pos_cand_inst_len'] = pos_inst_len
        batch['neg_cand_inst'] = neg_inst
        batch['neg_cand_inst_len'] = neg_inst_len
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

    def _get_neg_candidate_inst(self, device, exclude_idx):
        if (self.neg_candidate_inst is not None
            and self.neg_candidate_inst[0].device == device):
            inst, inst_len = self.neg_candidate_inst
        else:
            inst, inst_len = parse_batch_inst(
                self.inst_dict,
                range(self.args.num_pos_inst, self.inst_dict.num_insts),
                device)
            self.neg_candidate_inst = (inst, inst_len)

        # inst: [num_candidate, max_sentence_len]
        prob = np.ones((inst.size(0),), dtype=np.float32)

        for idx in exclude_idx:
            if idx == self.inst_dict.unknown_inst_idx:
                continue
            idx = idx - self.args.num_pos_inst
            if idx >= 0:
                prob[idx] = 0
        prob = prob / prob.sum()

        num_candidate = inst.size(0)
        select = np.random.choice(
            inst.size(0),
            self.args.num_neg_inst,
            replace=False,
            p=prob)
        select = torch.LongTensor(select).to(device)
        # select: [num_inst,]
        inst_len = inst_len.gather(0, select)
        select = select.unsqueeze(1).repeat(1, inst.size(1))
        inst = inst.gather(0, select)
        return inst, inst_len

    # ============ RL related ============
    def format_coach_input(self, batch, prefix=''):
        frame_passed = batch['frame_passed'].squeeze(1)
        frame_passed = frame_passed.clamp(max=self.max_instruction_span+1)
        data = {
            'prev_inst_idx': batch['prev_inst'].squeeze(1),
            'frame_passed': frame_passed,
            'count': batch[prefix+'count'],
            'base_count': batch[prefix+'base_count'],
            'cons_count': batch[prefix+'cons_count'],
            'moving_enemy_count': batch[prefix+'moving_enemy_count'],
        }
        # print(data['count'])
        # print(data['cons_count'])
        extra_data = self.glob_encoder.format_input(batch, prefix)
        data.update(extra_data)
        # print(data['prev_cmds'][0, :data['my_units']['num_units']])
        # print(data['map'][0].sum(2).sum(1))
        return data

    def _format_rl_language_input(self, batch):
        prev_inst, prev_inst_len = parse_batch_inst(
            self.inst_dict,
            batch['prev_inst_idx'].cpu().numpy(),
            batch['prev_inst_idx'].device)
        batch['prev_inst'] = prev_inst
        batch['prev_inst_len'] = prev_inst_len

        inst, inst_len = self._get_pos_candidate_inst(prev_inst.device)
        batch['cand_inst'] = inst
        batch['cand_inst_len'] = inst_len
        return batch

    def rl_forward(self, batch):
        """forward function use by RL
        """
        batch = self._format_rl_language_input(batch)
        glob_feat = self._forward(batch)
        v = self.value(glob_feat).squeeze()
        cont_prob = self.cont_cls.compute_prob(glob_feat)
        inst_prob = self.inst_selector.compute_prob(
            batch['cand_inst'], batch['cand_inst_len'], glob_feat)

        output = {
            'cont_pi': cont_prob,
            'inst_pi': inst_prob,
            'v': v
        }
        return output

    def sample(self, batch, word_based=True):
        """used for actor in ELF and visually evaulating model

        return
            inst: [batch, max_sentence_len], even inst is one-hot
            inst_len: [batch]
        """
        output = self.rl_forward(batch)
        samples = self.sampler.sample(
            output['cont_pi'], output['inst_pi'], batch['prev_inst_idx'])

        reply = {
            'cont': samples['cont'].unsqueeze(1),
            'cont_pi': output['cont_pi'],
            'inst': samples['inst'].unsqueeze(1),
            'inst_pi': output['inst_pi'],
        }

        # convert format needed by executor
        samples = []
        lengths = []
        raws = []
        for idx in reply['inst']:
            inst = self.inst_dict.get_inst(int(idx.item()))
            tokens, length = self.inst_dict.parse(inst, True)
            samples.append(tokens)
            lengths.append(length)
            raw = convert_to_raw_instruction(inst, self.max_raw_chars)
            raws.append(convert_to_raw_instruction(inst, self.max_raw_chars))

        device = reply['cont'].device
        if word_based:
            # for word based
            inst = torch.LongTensor(samples).to(device)
        else:
            inst = reply['inst']

        inst_len = torch.LongTensor(lengths).to(device)
        reply['raw_inst'] = torch.LongTensor(raws).to(device)
        return inst, inst_len, reply['cont'], reply
