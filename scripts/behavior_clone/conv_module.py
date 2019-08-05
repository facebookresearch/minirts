# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch.nn as nn
from torch.nn.utils import weight_norm

from common_utils import assert_eq, assert_lt
import common_utils.global_consts as gc


class ConvNet(nn.Module):
    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 num_hid_layers,
                 activate_out):
        super().__init__()

        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.out_channels = out_channels
        self.num_hid_layers = num_hid_layers
        self.activate_out = activate_out

        self.net = self._create_net()

    def _create_net(self):
        layers = []
        in_channels = self.in_channels
        for i in range(self.num_hid_layers):
            conv = nn.Conv2d(
                in_channels,
                self.hid_channels,
                3,
                padding=1)

            layers.append(weight_norm(conv))
            layers.append(nn.ReLU())
            in_channels = self.hid_channels

        conv = nn.Conv2d(
            in_channels,
            self.out_channels,
            3,
            padding=1)
        layers.append(weight_norm(conv))

        if self.activate_out:
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ConvFeatNet(nn.Module):
    def __init__(self,
                 in_dim,
                 hid_dim,
                 army_out_dim,
                 other_out_dim,
                 num_conv_layers,
                 num_post_layers):
        super().__init__()

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.army_out_dim = army_out_dim
        self.other_out_dim = other_out_dim

        # self.num_pre_layers = num_pre_layers
        self.num_conv_layers = num_conv_layers
        self.num_post_layers = num_post_layers

        # self.pre_net = self._create_conv1x1(in_dim, hid_dim, num_pre_layers)
        # if num_pre_layers > 0:
        #     in_dim = hid_dim

        assert num_conv_layers > 0
        # if num_conv_layers > 0:
        self.conv_net = ConvNet(
            in_dim, hid_dim, hid_dim, num_conv_layers - 1, True)
        # else:
        #     self.conv_net = self._no_net

        # self.feat_dim = self.conv_net.out_channels
        self.army_net = self._create_conv(hid_dim, army_out_dim, 1, 1, num_post_layers)
        self.enemy_net = self._create_conv(hid_dim, other_out_dim, 1, 1, num_post_layers)
        self.resource_net = self._create_conv(hid_dim, other_out_dim, 1, 1, num_post_layers)
        self.map_net = self._create_conv(hid_dim, other_out_dim, 1, 1, 1)

    @staticmethod
    def _no_net(x):
        return x

    @staticmethod
    def _create_conv(in_dim, out_dim, fsize, stride, depth):
        if depth == 0:
            return ConvFeatNet._no_net
        layers = []
        in_dim = in_dim
        for i in range(depth):
            conv = weight_norm(nn.Conv2d(in_dim, out_dim, fsize, stride=stride, padding=0))
            activate = nn.ReLU()
            layers.append(conv)
            layers.append(activate)
            in_dim = out_dim
        return nn.Sequential(*layers)

    @staticmethod
    def _select_feat_with_loc(feat, x, y):
        """select feature given locations

        feat: [batch, nc, h, w]
        x: [batch, pnum_unit], range: [0, 1)
        y: [batch, pnum_unit], range: [0, 1)

        return:
        selected_feat: [batch, pnum_unit, nc(feat_dim)]
        Note: returned feature is not masked at all
        """
        # feat: [batch, nc, h, w]
        x = x.long()
        y = y.long()
        if not x.max() < gc.MAP_X and x.min() >= 0:
            import pdb
            pdb.set_trace()

        assert(x.max() < gc.MAP_X and x.min() >= 0)
        assert(y.max() < gc.MAP_Y and y.min() >= 0)

        loc = y * gc.MAP_X + x
        # loc: [batch, pnum_unit]

        batch, nc, h, w = feat.size()
        assert_eq(h, gc.MAP_Y)
        assert_eq(w, gc.MAP_X)
        feat = feat.view(batch, nc, h * w)
        # feat: [batch, nc, h * w]

        loc = loc.unsqueeze(1).repeat(1, nc, 1)
        selected_feat = feat.gather(2, loc)
        # selected_feat: [batch, nc, pnum_unit]
        selected_feat = selected_feat.transpose(1, 2).contiguous()
        return selected_feat

    def forward(self, batch):
        # map_feat = self.pre_net(batch['map'])
        map_feat = self.conv_net(batch['map'])
        # army_feat = map_feat
        army_feat = self.army_net(map_feat)
        army_feat = self._select_feat_with_loc(
            army_feat,
            batch['my_units']['xs'],
            batch['my_units']['ys']
        )

        # # enemy_feat = map_feat
        # print(map_feat.size())
        # print(self.conv_net)
        # print(self.enemy_net)
        enemy_feat = self.enemy_net(map_feat)
        enemy_feat = self._select_feat_with_loc(
            enemy_feat,
            batch['enemy_units']['xs'],
            batch['enemy_units']['ys']
        )

        # resource_feat = map_feat
        resource_feat = self.resource_net(map_feat)
        resource_feat = self._select_feat_with_loc(
            resource_feat,
            batch['resource_units']['xs'],
            batch['resource_units']['ys']
        )

        map_feat = self.map_net(map_feat)
        return army_feat, enemy_feat, resource_feat, map_feat


def test_conv_feat_net():
    import pickle
    from dataset import BehaviorCloneDataset
    from torch.utils.data import DataLoader
    from module import ConvNet

    inst_dict = pickle.load(open('./data/new_inst_train.json_min10_dict.pt', 'rb'))
    inst_dict.set_max_sentence_length(20)

    dataset = BehaviorCloneDataset(
        './data/new_inst_dev.json_min10',
        11,
        50,
        10,
        inst_dict=inst_dict)
    loader = DataLoader(dataset, 10, shuffle=False, num_workers=0)
    loader = iter(loader)
    batch = next(loader)['current']

    conv_net = ConvNet(42, 32, 32, 2, True)
    conv_feat_net = ConvFeatNet(conv_net)
    army, enemy, resource, mapf = conv_feat_net(batch)
    for key, feat in [
            ('my_units', army),
            ('enemy_units', enemy),
            ('resource_units', resource)]:
        print(key, '-----')
        xs = batch[key]['xs']
        ys = batch[key]['ys']
        for i in range(xs.size(0)):
            pnum_units = xs.size(1)
            for uidx in range(pnum_units):
                x = xs[i][uidx]
                y = ys[i][uidx]
                feat_in_map = mapf[i, :, y, x]
                print('>>> feat in map: ', feat_in_map.size())
                feat_selected = feat[i][uidx]
                print('>>> feat in selected: ', feat_selected.size())

                diff = (feat_in_map - feat_selected).abs().max().item()
                print(diff)


def test_conv_feat_net2():
    import pickle
    from dataset import BehaviorCloneDataset
    from torch.utils.data import DataLoader
    from module import ConvNet
    import utils

    inst_dict = pickle.load(open('./data/new_inst_train.json_min10_dict.pt', 'rb'))
    inst_dict.set_max_sentence_length(20)

    dataset = BehaviorCloneDataset(
        './data/new_inst_valid.json_min10',
        11,
        50,
        10,
        inst_dict=inst_dict)
    dataset.data = dataset.data[10 * 4450:]

    loader = DataLoader(dataset, 10, shuffle=False, num_workers=0)

    device = torch.device('cuda:1')
    conv_net = ConvNet(42, 32, 32, 2, True)
    conv_feat_net = ConvFeatNet(conv_net).to(device)

    for i, batch in enumerate(loader):
        if i < 4450:
            continue
        print(i)
        utils.to_device(batch, device)
        # batch = batch['current']

        x = batch['my_units']['xs']
        y = batch['my_units']['ys']
        print(x)
        print(y)
        if x.max() >= gc.MAP_X:
            print(x)
            import pdb
            pdb.set_trace()
        if y.max() >= gc.MAP_Y:
            print(y)
            import pdb
            pdb.set_trace()

        x = batch['enemy_units']['xs']
        y = batch['enemy_units']['ys']
        print(x)
        print(y)
        if x.max() >= gc.MAP_X:
            print(x)
            import pdb
            pdb.set_trace()
        if y.max() >= gc.MAP_Y:
            print(y)
            import pdb
            pdb.set_trace()

        x = batch['resource_units']['xs']
        y = batch['resource_units']['ys']
        print(x)
        print(y)
        if x.max() >= gc.MAP_X:
            print(x)
            import pdb
            pdb.set_trace()
        if y.max() >= gc.MAP_Y:
            print(y)
            import pdb
            pdb.set_trace()

        conv_feat_net(batch)


if __name__ == '__main__':
    test_conv_feat_net2()
