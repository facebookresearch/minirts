# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import time
import json
from collections import defaultdict
from torch.utils.data import Dataset
import numpy as np

from common_utils import assert_eq, assert_neq
import common_utils.global_consts as gc


def merge_max_units(datasets):
    for key in datasets[0].max_num_units:
        val = 0
        for dataset in datasets:
            val = max(val, dataset.max_num_units[key])
        for dataset in datasets:
            dataset.max_num_units[key] = val


def append_prev_inst(data, num=5):
    insts = []
    ticks = []
    ref_replay = ''
    max_tick = 0
    for i, d in enumerate(data):
        replay = d['unique_id'].rsplit('-', 1)[0]
        if ref_replay != replay:
            ref_replay = replay
            insts = ['' for _ in range(num - 1)]
            ticks = [i for _ in range(num - 1)]
        # print(insts)
        if d['instruction'] != insts[-1]:
            insts.append(d['instruction'])
            ticks.append(i)
        d['hist_inst'] = insts[-num:]
        hist_tick = ticks[-num:]
        tick_diff = [(i - j) // 5 for j in hist_tick]
        # print(insts[-num:])
        # print(tick_diff)
        # print('----------')
        max_tick = max(max_tick, np.max(tick_diff))
        d['hist_inst_diff'] = tick_diff

    print('max tick diff:', max_tick)
    return data


class BehaviorCloneDataset(Dataset):
    def __init__(self,
                 json_file,
                 num_resource_bin,
                 resource_bin_size,
                 max_num_prev_cmds,
                 *,
                 inst_dict=None,
                 word_based=True,
                 data=None):
        if data is not None:
            self.data = data
        else:
            print('loading json from:', json_file)
            self.data = json.loads(open(json_file).read())
            print('finish loading json, %d entry loaded' % len(self.data))

        self.data = append_prev_inst(self.data)

        self.max_num_units = self._get_dataset_stats(json_file)
        self.num_resource_bin = num_resource_bin
        self.resource_bin_size = resource_bin_size
        self.max_num_prev_cmds = max_num_prev_cmds
        self.inst_dict = inst_dict
        self.word_based = word_based

        self.cache = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.cache is not None:
            return self.cache[index]

        entry = self.data[index]
        data = self._process_entry(entry)
        return data

    def _get_dataset_stats(self, json_file):
        stats_file = json_file + '-stats.json'
        if (os.path.exists(stats_file)
            and os.path.getmtime(stats_file) >= os.path.getmtime(json_file)):
            print('loading dataset stats from ', stats_file)
            max_num_units = json.load(open(stats_file))
        else:
            print('recomputing dataset stats')
            max_num_units = self._compute_max_num_units(
                ['my_units', 'enemy_units', 'resource_units']
            )
            json.dump(max_num_units, open(stats_file, 'w'), indent=4)
        return max_num_units

    def _compute_max_num_units(self, unit_keys):
        max_num_units = {key : 0 for key in unit_keys}
        for d in self.data:
            for key in unit_keys:
                num_units = len(d[key])
                max_num_units[key] = max(max_num_units[key], num_units)
        return max_num_units

    def _process_entry(self, entry):
        inst, inst_len, inst_idx = self._process_instruction(entry['instruction'])
        hist_inst, hist_inst_len, hist_inst_idx = \
            self._process_hist_instruction(entry['hist_inst'])
        resource_bin = self._process_resource(entry)

        max_my_units = self.max_num_units['my_units']
        my_units = entry['my_units']
        my_units_info = self._process_units(my_units, max_my_units)
        current_cmds = self._process_cmds(my_units, 'current_cmd', max_my_units)
        target_cmds = self._process_cmds(my_units, 'target_cmd', max_my_units)
        prev_cmds = self._process_prev_cmds(
            'prev_cmd',
            my_units,
            self.max_num_prev_cmds,
            0,
            max_my_units)

        enemy_units_info = self._process_units(
            entry['enemy_units'], self.max_num_units['enemy_units'])
        resource_units_info = self._process_units(
            entry['resource_units'], self.max_num_units['resource_units'])

        data_point = {
            'inst': inst if self.word_based else inst_idx,
            'inst_len': inst_len,
            'hist_inst': hist_inst if self.word_based else hist_inst_idx,
            'hist_inst_len': hist_inst_len,
            'hist_inst_diff': np.array(entry['hist_inst_diff'], dtype=np.int64),
            'resource_bin': resource_bin,
            'my_units': my_units_info,
            'enemy_units': enemy_units_info,
            'resource_units': resource_units_info,
            'prev_cmds': prev_cmds,
            'current_cmds': current_cmds,
            'map': self._process_map(entry),
            'target_cmds': target_cmds,
            'glob_cont': np.int64(entry['glob_cont']),
        }
        return data_point

    def _process_map(self, entry):
        coord_offset = 0
        visibility_offset = coord_offset + 2
        terrain_offset = visibility_offset + len(gc.Visibility)
        army_offset = terrain_offset + len(gc.Terrain)
        enemy_offset = army_offset + len(gc.UnitTypes) - 1
        resource_offset = enemy_offset + len(gc.UnitTypes) - 1
        num_channels = resource_offset + 1

        map_tensor = np.zeros(
            (num_channels, gc.MAP_Y, gc.MAP_X),
            dtype=np.float32)

        # fill in visibility and terrain
        visibility = entry['map']['visibility']
        terrain = entry['map']['terrain']
        for y in range(gc.MAP_Y):
            for x in range(gc.MAP_X):
                loc = y * gc.MAP_X + x
                v = visibility[loc]
                t = terrain[loc]
                assert(v < len(gc.Visibility))
                assert(t < len(gc.Terrain))
                map_tensor[0][y][x] = y / gc.MAP_Y
                map_tensor[1][y][x] = x / gc.MAP_X
                map_tensor[visibility_offset + v][y][x] = 1
                map_tensor[terrain_offset + t][y][x] = 1

        offset_units = [
            (army_offset, entry['my_units']),
            (enemy_offset, entry['enemy_units']),
            (resource_offset, entry['resource_units'])
        ]
        for offset, units in offset_units:
            for unit in units:
                type_offset = offset + unit['unit_type']
                # subtract resource for army and enemy
                if offset != resource_offset:
                    type_offset -= 1
                y = int(unit['y'])
                x = int(unit['x'])
                map_tensor[type_offset][y][x] += 1

        return map_tensor

    def _process_prev_cmds(self,
                           key,
                           my_units,
                           max_history_len,
                           padding_idx,
                           max_num_units):
        cmds = []
        for unit in my_units:
            prev_cmds = unit[key]

            real_cmds = []
            for cmd in prev_cmds:
                # cmd_type = cmd['cmd_type']
                cmd_type = cmd#['cmd_type']
                # assert_neq(cmd_type, gc.CmdTypes.IDLE.value)
                # assert_neq(cmd_type, gc.CmdTypes.CONT.value)
                real_cmds.append(cmd_type)

            num_real_cmds = len(real_cmds)
            if num_real_cmds > max_history_len:
                # print('warning: exceed max history len:', num_real_cmds)
                # print(real_cmds)
                real_cmds = real_cmds[-max_history_len:]

            assert len(real_cmds) <= max_history_len
            # num_cmds.append(num_real_cmds)
            pad = [padding_idx for _ in range(len(real_cmds), max_history_len)]
            real_cmds = pad + real_cmds
            assert len(real_cmds) == max_history_len
            # real_cmds.extend(
            #     [padding_idx for _ in range(num_real_cmds, max_history_len)]
            # )
            cmds.append(real_cmds)

        cmds = np.array(cmds, dtype=np.int64)
        cmds = self._pad(cmds, max_num_units, 0)
        # num_cmds = np.array(num_cmds)
        # num_cmds = self._pad(num_cmds, max_num_units, 0)
        return cmds

    def _pad(self, array, padded_size, padding_idx):
        shape = array.shape
        if len(array.shape) == 1:
            pad_shape = (padded_size - array.shape[0])
            stack_func = np.hstack
        elif len(array.shape) == 2:
            pad_shape = (padded_size - array.shape[0], array.shape[1])
            stack_func = np.vstack
        else:
            assert False, 'Num dims %d not supported'

        pad = np.zeros(pad_shape, dtype=array.dtype) + padding_idx
        padded_array = stack_func((array, pad))
        return padded_array

    def _process_hist_instruction(self, hist_insts):
        insts = []
        ls = []
        idxs = []
        for inst_str in hist_insts:
            inst, l, idx = self._process_instruction(inst_str)
            insts.append(inst)
            ls.append(l)
            idxs.append(idx)

        insts = np.array(insts, dtype=np.int64)
        ls = np.array(ls, dtype=np.int64)
        idxs = np.array(idxs, dtype=np.int64)
        return insts, ls, idxs

    def _process_instruction(self, ins):
        if self.inst_dict is not None:
            instruction, length = self.inst_dict.parse(ins, True)
            inst_idx = self.inst_dict.get_inst_idx(ins)
            return np.array(instruction, dtype=np.int64), length, inst_idx
        else:
            assert False

    def _process_resource(self, entry, field='resource'):
        resource_bin = np.zeros(self.num_resource_bin, dtype=np.float32)
        resource = entry[field]
        bin_idx = min(resource // self.resource_bin_size,
                      self.num_resource_bin - 1)
        resource_bin[bin_idx] = 1
        return resource_bin

    def _process_units(self, units, padded_size):
        types = []
        xs = []
        ys = []
        hps = []
        for u in units:
            x = u['x']
            y = u['y']
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            types.append(u['unit_type'])
            xs.append(x)
            ys.append(y)
            hps.append(u['hp'])

        types = np.array(types, dtype=np.int64)
        xs = np.array(xs, dtype=np.int64)
        ys = np.array(ys, dtype=np.int64)
        hps = np.array(hps, dtype=np.int64)
        num_units = len(units)
        assert_eq(num_units, types.shape[0])

        types = self._pad(types, padded_size, 0)
        xs = self._pad(xs, padded_size, 0)
        ys = self._pad(ys, padded_size, 0)
        hps = self._pad(hps, padded_size, 0)

        units_info = {
            'types': types,
            'xs': xs,
            'ys': ys,
            'hps': hps,
            'num_units': num_units
        }
        return units_info

    def _process_cmds(self, units, cmd_name, padded_size):
        """

        units: (list) of our/enemy/resource units
        cmd_name: (str) current_cmd or target_cmd
        padded_size: padded num of units
        """
        field_and_pad_idxs = [
            ('cmd_type', 0),
            ('target_type', 0),
            ('target_x', 0),
            ('target_y', 0),
            ('target_gather_idx', 0),
            ('target_attack_idx', 0)
        ]

        data = defaultdict(list)
        for u in units:
            if cmd_name in u:
                cmd = u[cmd_name]
            else:
                cmd = None

            for field, pad_idx in field_and_pad_idxs:
                if cmd is None:
                    val = pad_idx
                else:
                    assert(cmd[field] != -1)
                    val = cmd[field]
                    if field == 'target_x' or field == 'target_y':
                        val = round(val)
                data[field].append(val)

        for field, pad_idx in field_and_pad_idxs:
            tensor = np.array(data[field], dtype=np.int64)
            data[field] = self._pad(tensor, padded_size, pad_idx)

        return data


def test_process_map():
    from torch.utils.data import DataLoader

    inst_dict = pickle.load(open('./data/new_inst_train.json_min10_dict.pt', 'rb'))
    inst_dict.set_max_sentence_length(20)

    dataset = BehaviorCloneDataset(
        './data/new_inst_valid.json_min10',
        11,
        50,
        10,
        inst_dict=inst_dict)
    loader = DataLoader(dataset, 1000, shuffle=False, num_workers=0)
    loader = iter(loader)
    batch = next(loader)['current']

    map_tensor = batch['map']

    # copied, TODO: should make this property?
    coord_offset = 0
    visibility_offset = coord_offset + 2
    terrain_offset = visibility_offset + len(gc.Visibility)
    army_offset = terrain_offset + len(gc.Terrain)
    enemy_offset = army_offset + len(gc.UnitTypes) - 1
    resource_offset = enemy_offset + len(gc.UnitTypes) - 1
    num_channels = resource_offset + 1

    offset_units = [
        (army_offset, batch['my_units']),
        (enemy_offset, batch['enemy_units']),
        (resource_offset, batch['resource_units']),
    ]

    for i in range(map_tensor.size(0)):
        for offset, units in offset_units:
            num_units = units['num_units'][i].item()
            for uidx in range(num_units):
                unit_type = units['types'][i][uidx].item()
                type_offset = offset + unit_type
                if offset != resource_offset:
                    type_offset -= 1

                unit_type = gc.UnitTypes(unit_type).name
                print('type: %s; offset: %s' % (unit_type, type_offset))
                x = units['xs'][i][uidx]
                y = units['ys'][i][uidx]
                assert(map_tensor[i][type_offset][y][x] > 0)


if __name__ == '__main__':
    import pickle
    # test_process_map()

    inst_dict = pickle.load(open(
        '/private/home/hengyuan/scratch/rts_data/dataset3_ppcmd2/dict.pt', 'rb'))
    inst_dict.set_max_sentence_length(20)

    dataset = BehaviorCloneDataset(
        '/private/home/hengyuan/scratch/rts_data/dataset3_nofow/dev.json',
        11,
        50,
        20,
        inst_dict=inst_dict)

    # prev_inst = ''
    # for i, entry in enumerate(dataset.data):
    #     if entry['base_frame_idx'] == i:
    #         print(i)
    #         print(prev_inst)
    #         print(entry['instruction'])
    #         prev_inst = entry['instruction']

    # from utils import analyze_build
    # analyze_build(dataset)

    # from utils import write_tensor_to_image
    # folder = 'test_map'
    # for i, entry in enumerate(dataset.data):
    #     tensor = dataset._process_map(entry)
    #     filename = os.path.join(folder, '%s.png' % (str(i).zfill(5)))
    #     write_tensor_to_image(tensor, filename)


    # dataset = BehaviorCloneDataset(
    #     './data/fake-valid.json',
    #     11,
    #     50,
    #     10,
    #     inst_dict=None,
    # )
