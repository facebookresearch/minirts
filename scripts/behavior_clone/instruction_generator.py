# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class LanguageGenerator(nn.Module):
    def __init__(self, cmd_dict, word_emb, emb_dim, context_dim, hid_dim, out_dropout):
        super().__init__()

        self.cmd_dict = cmd_dict
        self.word_emb = word_emb

        self.reader = nn.LSTM(emb_dim + context_dim, hid_dim, batch_first=True)
        self.writer = nn.LSTMCell(emb_dim + context_dim, hid_dim)
        # tie the weights of reader and writer
        self.writer.weight_ih = self.reader.weight_ih_l0
        self.writer.weight_hh = self.reader.weight_hh_l0
        self.writer.bias_ih = self.reader.bias_ih_l0
        self.writer.bias_hh = self.reader.bias_hh_l0

        self.output_dropout = nn.Dropout(out_dropout)
        self.decoder = nn.Linear(hid_dim, cmd_dict.vocab_size)

    def compute_loss(self, x, y, context):
        """
        x: [batch, max_length]
        y: [batch, max_length]
        context: [batch, context_dim], used to initialize context state
        """
        # print('context:', context.size())
        emb = self.word_emb(x)
        # emb: [batch, max_length, emb_dim]
        context = context.unsqueeze(1).repeat(1, emb.size(1), 1)

        input_ = torch.cat([emb, context], 2)
        # cell = torch.zeros(hidden.size(), device=hidden.device)
        output, _ = self.reader(input_)
        output = self.output_dropout(output)
        logit = self.decoder(output)
        # logit: [batch, length, num_words]
        logp = nn.functional.log_softmax(logit, 2)
        logp = logp.gather(2, y.unsqueeze(2)).squeeze(2)
        loss = -logp.sum(1)
        return loss

    def sample(self, context, temperature):
        batchsize = context.size(0)
        max_len = self.cmd_dict.max_num_words
        pad = self.cmd_dict.get_words_pad()
        sentence = torch.zeros(batchsize, max_len + 1, device=context.device)
        sentence = (sentence + pad).long()
        sentence[:, 0] = self.cmd_dict.get_word_idx('<start>')

        idx = 1
        while idx <= self.cmd_dict.max_num_words:
            word_emb = self.word_emb(sentence[:, idx - 1])
            # print('word_emb:', word_emb.size())
            # print('context:', context.size())
            input_ = torch.cat([word_emb, context], 1)
            if idx == 1:
                hidden, cell = self.writer(input_)
            else:
                hidden, cell = self.writer(input_, (hidden, cell))

            logit = self.decoder(self.output_dropout(hidden))
            weight = logit.div(temperature).exp().cpu()
            word = torch.multinomial(weight, 1).squeeze(1)
            sentence[:, idx] = word
            idx += 1

        sentence = sentence[:, 1:]
        sentence_len = torch.zeros((batchsize,), device=context.device)
        sentence_len = (sentence_len + max_len).long()
        for i in range(batchsize):
            for j in range(sentence.size(1)):
                if sentence[i][j] == pad:
                    sentence_len[i] = j
                    break

        return sentence, sentence_len


class RnnLanguageGenerator(nn.Module):
    def __init__(self,
                 word_emb,
                 word_emb_dim,
                 context_dim,
                 hid_dim,
                 vocab_size,
                 inst_dict,
    ):
        super().__init__()

        self.word_emb = word_emb
        self.vocab_size = vocab_size
        self.inst_dict = inst_dict

        self.rnn = nn.LSTM(word_emb_dim + context_dim, hid_dim, batch_first=True)
        self.decoder = weight_norm(nn.Linear(hid_dim, vocab_size), dim=None)

    def _forward2d(self, x, context):
        """compute logp given input

        args:
            x: [batch, max_len]
            context: [batch, context_dim]

        return:
            logp: [batch, max_len, vocab_size]
        """
        emb = self.word_emb(x)
        context = context.unsqueeze(1).repeat(1, x.size(1), 1)
        input_ = torch.cat([emb, context], 2)
        output, _ = self.rnn(input_)
        logit = self.decoder(output)
        logp = nn.functional.log_softmax(logit, 2)
        return logp

    def _forward3d(self, x, context):
        """compute logp given input

        args:
            x: [num_inst, max_len}
            context: [batch, context_dim]

        return:
            logp: [batch, num_inst, max_len, vocab_size]
        """
        emb = self.word_emb(x)
        emb = emb.unsqueeze(0).repeat(context.size(0), 1, 1, 1)
        context = context.unsqueeze(1).repeat(1, x.size(1), 1)
        logit = []
        for i in range(x.size(0)):
            input_ = torch.cat([emb[:, i], context], 2)
            output, _ = self.rnn(input_)
            logit.append(self.decoder(output))

        logit = torch.stack(logit, 1)
        logp = nn.functional.log_softmax(logit, 3)
        return logp

    def compute_loss(self, x, y, context):
        """compute nll loss

        args:
            x: [batch, max_length]
            y: [batch, max_length]
            context: [batch, context_dim], used at every time step
        return:
            nll: [batch], -logp
        """
        logp = self._forward2d(x, context)
        logp = logp.gather(2, y.unsqueeze(2)).squeeze(2)
        logp = logp.sum(1)
        return -logp

    def compute_prob(self, x, y, context, *, temp=1, log=False):
        """compute log prob of target y given input x

        args:
            x: [num_inst, max_length]
            y: [num_inst, max_length]
            context: [batch, context_dim], used at every time step
        return:
            logp: [batch, num_instruction]
        """
        logp = self._forward3d(x, context)
        # logp: [batch, num_inst, max_length, vocab_size]
        batch, num_inst, max_len, vocab_size = logp.size()
        # print(y.size())
        y = y.unsqueeze(0).expand(context.size(0), y.size(0), y.size(1))
        logp = logp.gather(3, y.unsqueeze(3)).squeeze(3)
        # logp:  [batch, num_inst, max_length]
        logit = logp.sum(2)
        logit = logit / temp
        # logit: [batch, num_inst]
        if log:
            return nn.functional.log_softmax(logit, 1)
        else:
            return nn.functional.softmax(logit, 1)

    def compute_prob2(self, x, y, context):
        """compute logp given input

        args:
            x: [num_inst, max_len]
            context: [batch, context_dim]
        """
        emb = self.word_emb(x)
        # emb: [num_inst, emb_dim]
        # logit = []
        # logp = []
        logps = []
        for i in range(context.size(0)):
            context_  = context[i].unsqueeze(0).unsqueeze(1).repeat(
                emb.size(0), emb.size(1), 1)
            # print(emb.size())
            # print(context_.size())
            input_ = torch.cat([emb, context_], 2)
            output, _ = self.rnn(input_)
            # logit.append(self.decoder(output))
            logit = self.decoder(output)
            # logit: [num_inst, max_len, vocab_size]
            logp = nn.functional.log_softmax(logit, 2)#.sum(2)
            # print(logp.size())
            # print(y[i].size())
            logp = logp.gather(2, y.unsqueeze(2)).squeeze(2)
            logp = logp.sum(1)
            logps.append(logp)

        # logit = torch.stack(logit, 0)
        # print('>>>', logit.size())
        # logp = nn.functional.log_softmax(logit, 3)
        logps = torch.stack(logps, 0)
        print(logps.size())
        logps = nn.functional.softmax(logps, 1)
        return logps
