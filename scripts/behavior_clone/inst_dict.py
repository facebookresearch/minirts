# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import re


def clean_instruction(cmd):
    cmd = cmd.strip().lower()
    cmd = re.sub(r'[;,:.!?&()-/]', ' ', cmd)
    return cmd


def correct_instruction(inst, corrections):
    inst = clean_instruction(inst)
    words = inst.split()
    new_words = []
    for w in words:
        w = w.strip()
        if w in corrections:
            new_words.append(corrections[w])
        else:
            new_words.append(w)
    inst = ' '.join(new_words)
    return inst


class InstructionDict:
    def __init__(self, word_counts, inst_counts, corrections):
        self._word_counts = word_counts
        self._inst_counts = inst_counts

        self._word2idx = {}
        self._idx2word = []
        self._inst2idx = {}
        self._idx2inst = []

        # sort by frequency in decreasing order
        for word in sorted(word_counts.keys(), key=lambda w: -word_counts[w]):
            self._add_word(word)

        # sort by frequency in decreasing order
        for inst in sorted(inst_counts.keys(), key=lambda c: -inst_counts[c]):
            self._add_inst(inst)

        self._corrections = corrections
        self._max_sentence_length = None

    @property
    def vocab_size(self):
        return len(self._word2idx)

    @property
    def total_vocab_size(self):
        """vocab_size + <unk>, <pad>, <start>"""
        return self.vocab_size + 3

    @property
    def num_insts(self):
        return len(self._inst2idx)

    @property
    def total_num_insts(self):
        """num_insts + <unk>, <pad>"""
        return self.num_insts + 2

    @property
    def unknown_inst_idx(self):
        return len(self._inst2idx)

    @property
    def pad_inst_idx(self):
        return len(self._inst2idx) + 1

    @property
    def unknown_word_idx(self):
        return self.vocab_size

    @property
    def pad_word_idx(self):
        return self.vocab_size + 1

    @property
    def start_word_idx(self):
        return self.vocab_size + 2

    def set_max_sentence_length(self, max_sentence_length):
        self._max_sentence_length = max_sentence_length

    def _add_word(self, word):
        self._word2idx[word] = len(self._idx2word)
        self._idx2word.append(word)

    def _add_inst(self, inst):
        self._inst2idx[inst] = len(self._idx2inst)
        self._idx2inst.append(inst)

    def put_forward_inst(self, insts):
        all_inst = insts + self._idx2inst
        inst2idx = {}
        idx2inst = []
        for inst in all_inst:
            if inst not in inst2idx:
                inst2idx[inst] = len(idx2inst)
                idx2inst.append(inst)
            else:
                assert inst in insts

        assert len(inst2idx) == len(self._inst2idx)
        self._inst2idx = inst2idx
        self._idx2inst = idx2inst

    def get_word_idx(self, word):
        idx = self._word2idx.get(word, self.unknown_word_idx)
        return idx

    def get_inst_idx(self, inst):
        """return a number in [0, _max_num_instructions) or unknown_inst_idx
        """
        # if len(inst) == 0:
        #     return self.pad_inst_idx

        idx = self._inst2idx.get(inst, None)
        if idx is not None:
            assert(self._idx2inst[idx] == inst)

        if idx is None:
            idx = self.unknown_inst_idx
        return idx

    def get_inst(self, idx):
        # assert idx >= 0 and idx <= self.total_num_insts
        if idx == -1:
            idx = self.pad_inst_idx

        if idx < self.num_insts:
            return self._idx2inst[idx]
        elif idx == self.pad_inst_idx:
            return ''
        elif idx == self.unknown_inst_idx:
            return '<unk>'
        else:
            assert False

    def parse(self, inst, should_pad):
        inst = correct_instruction(inst, self._corrections)
        if len(inst) == 0:
            words = []
        else:
            words = inst.split(' ')[ : self._max_sentence_length]
        length = min(len(words), self._max_sentence_length - 1)

        tokens = []
        # -1 to make sure there will be at least one <pad> (<end>)
        for word in words[ : self._max_sentence_length - 1]:
            tokens.append(self.get_word_idx(word))

        while should_pad and len(tokens) < self._max_sentence_length:
            tokens.append(self.pad_word_idx)

        return tokens, length

    def deparse(self, inst):
        """convert tensor to natural language instruction

        tokens: [batch, max_sentence_length]
        """
        assert inst.dim() == 2
        inst = inst.cpu()
        sentence = []
        for ins in inst:
            lang = []
            for token in ins:
                if token == self.pad_word_idx:
                    break
                lang.append(self._idx2word[token])

            sentence.append(' '.join(lang))
        return sentence


# def put_forward_scout_inst(inst_dict):
#     scout_inst = [
#         'send idle peasant to scout for enemy', # 253
#         'scout for resources', # 271
#         'send one peasant to scout', # 277
#         'send 2 peasant to mine 1 scout', # 335
#         'send idle peasant to scout', # 361
#         'scout for mineral', # 363,
#         'send one peasant to scout the map', # 375
#         'send one peasant to scout for enemy', # 385
#         'scout for enemy base with idle peasant', # 453
#         'send idle peasant to scout for enemy base', # 466
#         'scout north', # 487
#         'send one peasant to scout for enemies', # 542
#         'use idle peasant to scout enemy base retreat when found', # 556
#         'scout map with idle worker', # 571
#         'send peasant to scout', # 613
#         'scout with idle peasant', # 636
#         'scout west', # 680
#         'scout south', # 713
#         'train peasant to scout for enemy base', # 721
#         # 'send one peasant to scout for enemy base', # 725,
#         'send idle peasant to scout enemy base', # 732,
#         # 'send idle peasant to scout the map', # 742
#         # 'mine with 2 peasant and scout with the 3rd', # 869
#         'train peasant scout map', # 901
#     ]
#     inst_dict.put_forward_inst(scout_inst)
#     return inst_dict


# def convert_simple_dict(inst_dict):
#     inst_dict.inst2idx = {}
#     inst_dict.idx2inst = []

#     inst_dict.add_inst('<unk>')
#     inst_dict.add_inst('<pad>')

#     inst_dict.add_inst('build a blacksmith')
#     inst_dict.add_inst('build a stable')
#     inst_dict.add_inst('build a barrack')
#     inst_dict.add_inst('build a workshop')
#     inst_dict.add_inst('build a guard tower')
#     inst_dict.add_inst('build swordman')
#     inst_dict.add_inst('build cavalry')
#     inst_dict.add_inst('build spearman')
#     inst_dict.add_inst('build dragon')
#     inst_dict.add_inst('build catapult')
#     inst_dict.add_inst('build archer')
#     inst_dict.add_inst('build peasant')
#     inst_dict.add_inst('send all peasant to mine')
#     inst_dict.add_inst('send idle peasant to mine')
#     inst_dict.add_inst('mine resource')
#     inst_dict.add_inst('scout the map')
#     inst_dict.add_inst('find enemy')
#     inst_dict.add_inst('attack')
#     inst_dict.add_inst('attack enemy')
#     return inst_dict
