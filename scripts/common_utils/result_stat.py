# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Stats for game result (win/loss) distribution

Note that in minirts, draw (exceed max_num_tick) means loss for trained agent
so draw is not counted in this impl
"""
import os
import numpy as np
from tensorboardX import SummaryWriter


class ResultStat:
    def __init__(self, batch_key, root):
        self.batch_key = batch_key
        self.win = 0
        self.loss = 0
        self.tie = 0
        if root is not None:
            self.tb_writer = SummaryWriter(os.path.join(root, 'result.tb'))
        else:
            self.tb_writer = None

    @property
    def num_games(self):
        return self.win + self.loss

    def reset(self):
        self.win = 0
        self.loss = 0

    def feed(self, batch):
        rewards = batch[self.batch_key]
        rewards = rewards.cpu().numpy()
        unique, counts = np.unique(rewards, return_counts=True)
        counts = dict(zip(unique, counts))
        self.win += counts.get(1, 0)
        self.loss += counts.get(-1, 0)

    def log(self, epoch):
        if self.num_games == 0:
            win_rate = 0
        else:
            win_rate = self.win / self.num_games

        if self.tb_writer is not None:
            self.tb_writer.add_scalar('winrate', win_rate, epoch)

        msg = 'win: {0:4d}, loss: {1:4d}, winrate: {2:.2%}'.format(
            self.win, self.loss, win_rate)
        return msg
