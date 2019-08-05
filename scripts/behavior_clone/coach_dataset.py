# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import json
import time

from torch.utils.data import Dataset, DataLoader
import numpy as np

from dataset import BehaviorCloneDataset
from common_utils import assert_eq, assert_neq
import common_utils.global_consts as gc


def compute_cache(dataset, num_workers=40):
    t = time.time()
    cache =  []
    loader = DataLoader(dataset,
                        num_workers * 5 if num_workers > 0 else 1,
                        shuffle=False,
                        num_workers=num_workers,
                        collate_fn=lambda x : x)
    for i, batch in enumerate(loader):
        for data in batch:
            cache.append(data)

    dataset.cache = cache
    print('time taken to compute cache: %.2f' % (time.time() - t))


class CoachDataset(Dataset):
    def __init__(self,
                 json_file,
                 moving_avg_decay,
                 num_resource_bin,
                 resource_bin_size,
                 max_num_prev_cmds,
                 inst_dict,
                 max_instruction_span,
                 *,
                 num_instructions=-1,
                 remove_odd_inst=False,
                 remove_even_inst=False):
        print('loading json from:', json_file)
        self.data = json.loads(open(json_file).read())
        print('finish loading json, %d entry loaded' % len(self.data))

        self.executor_dataset = BehaviorCloneDataset(
            json_file,
            num_resource_bin,
            resource_bin_size,
            max_num_prev_cmds,
            inst_dict=inst_dict,
            data=self.data)

        self.moving_avg_decay = moving_avg_decay
        self.num_resource_bin = num_resource_bin
        self.resource_bin_size = resource_bin_size
        self.max_num_prev_cmds = max_num_prev_cmds
        self.inst_dict = inst_dict
        self.max_instruction_span = max_instruction_span

        self.idx_mapping = self._filter_on_instruction_span()
        # self.filtered = True

        self.num_instructions = num_instructions
        if num_instructions > 0:
            self.idx_mapping = self._filter_with_instruction()

        if remove_odd_inst or remove_even_inst:
            assert not (remove_odd_inst and remove_even_inst)
            assert num_instructions <= 0
            self.idx_mapping = self._filter_with_odd_even(remove_odd_inst, remove_even_inst)

        self.cache = None

    @staticmethod
    def get_num_count_channels():
        # keep in sync with _process_count
        c = len(gc.Visibility) + (len(gc.UnitTypes) - 1) + (len(gc.UnitTypes) - 1) + 1
        return c

    def __len__(self):
        # if self.filtered:
        return len(self.idx_mapping)
        # else:
        #     return len(self.data)

    def __getitem__(self, index):
        if self.cache is not None:
            return self.cache[index]

        # if self.filtered:
        index = self.idx_mapping[index]

        entry = self.data[index]

        data = self._process_entry(index, entry)
        data['is_base_frame'] = int(index == entry['executor_base_frame_idx'])

        data2 = self.executor_dataset[index]
        data2.update(data)
        return data2

    def _filter_on_instruction_span(self):
        print('filtering on instruction span >=', self.max_instruction_span)
        idx_mapping = []
        avg_span = []
        new_avg_span = []
        i = 0
        while i < len(self.data):
            # assert(self.data[i]['coach_base_frame_idx'] == i)
            j = i
            while j < len(self.data) \
                  and self.data[j]['instruction'] == self.data[i]['instruction']:
                j += 1
            span = j - i
            avg_span.append(span)
            if span < self.max_instruction_span:
                new_avg_span.append(span)
                k = i
                while k < j:
                    idx_mapping.append(k)
                    k += 1
            i = j
        print('before filtering:', len(self.data), ', avg span: %.2f' % np.mean(avg_span))
        print('after filtering:', len(idx_mapping), ', avg span: %.2f' % np.mean(new_avg_span))
        return idx_mapping

    def _filter_with_instruction(self):
        print('filtering out frames with instruction idx >', self.num_instructions)
        idx_mapping = []
        for i in self.idx_mapping:
            d = self.data[i]
            idx = self.inst_dict.get_inst_idx(d['instruction'])

            if idx < self.num_instructions:
                idx_mapping.append(i)

        print('before filtering:', len(self.idx_mapping))
        print('after filtering:', len(idx_mapping))
        return idx_mapping

    def _filter_with_odd_even(self, remove_odd, remove_even):
        if remove_odd:
            print('filtering out ODD frames')
        else:
            print('filtering out EVEN frames')
        idx_mapping = []
        for i in self.idx_mapping:
            d = self.data[i]
            idx = self.inst_dict.get_inst_idx(d['instruction'])
            if idx == self.inst_dict.unknown_inst_idx:
                continue
            if remove_odd and idx % 2 == 1:
                continue
            if remove_even and idx % 2 == 0:
                continue
            # print(idx)
            idx_mapping.append(i)

        print('before filtering:', len(self.idx_mapping))
        print('after filtering:', len(idx_mapping))
        return idx_mapping

    def _get_common_features(self, index, entry):
        """shared code by _process_word_entry and _process_one_hot_entry
        """
        count = self._process_count(entry)
        base_count = self._process_count(self.data[entry['coach_base_frame_idx']])
        cons_count = np.array(entry['cons_count'], dtype=np.float32)
        moving_enemy_count = self._compute_moving_avg_enemy_count(index)

        frame_passed = index - entry['coach_base_frame_idx']
        # print(frame_passed)
        # if frame_passed >= self.max_instruction_span:
        #     print('bad frame_passed:', frame_passed,
        #           ', should be <', self.max_instruction_span)
        #     assert False
        if frame_passed > self.max_instruction_span:
            frame_passed = self.max_instruction_span + 1

        data_point = {
            'count': count,
            'base_count': base_count,
            'cons_count': cons_count,
            'moving_enemy_count': moving_enemy_count,
            'resource_bin': self._process_resource(entry),
            'frame_passed': frame_passed,
        }
        return data_point

    def _process_entry(self, index, entry):
        prev_inst, prev_inst_len = self._process_instruction(entry['prev_instruction'])
        inst, inst_len = self._process_instruction(entry['instruction'])
        inst_idx = self.inst_dict.get_inst_idx(entry['instruction'])
        data_point = self._get_common_features(index, entry)

        data_point['prev_inst'] = prev_inst
        data_point['prev_inst_len'] = prev_inst_len
        data_point['inst'] = inst
        data_point['inst_len'] = inst_len
        data_point['inst_idx'] = inst_idx
        return data_point

    def _compute_moving_avg_enemy_count(self, index):
        avg_count = None
        ref_replay_name = self.data[index]['unique_id'].rsplit('-', 1)[0]
        i = index
        weight = 1 - self.moving_avg_decay
        while i >= 0:
            replay_name = self.data[i]['unique_id'].rsplit('-', 1)[0]
            if replay_name != ref_replay_name:
                break

            count = self._process_count(self.data[i], enemy_only=True)
            # add resource, for the ease of c++ impl
            count = np.hstack(((0,), count))
            # print(count)
            if avg_count is None:
                avg_count = count * weight
            else:
                avg_count += count * weight
            # print(avg_count)

            i -= 1
            weight *= self.moving_avg_decay
        # print('=====')
        return avg_count.astype(np.float32)

    def _process_resource(self, entry, field='resource'):
        resource_bin = np.zeros(self.num_resource_bin, dtype=np.float32)
        resource = entry[field]
        bin_idx = min(resource // self.resource_bin_size,
                      self.num_resource_bin - 1)
        resource_bin[bin_idx] = 1
        return resource_bin

    def _process_unit_type(self, units, padded_size):
        unit_types = np.zeros((padded_size,), dtype=np.int64)
        for i, u in enumerate(units):
            unit_types[i] = u['unit_type']

        return unit_types, len(units)

    # def _process_current_cmd(self, units, padded_size):
    #     current_cmd_type = np.zeros((padded_size,), dtype=np.int64)
    #     current_cmd_cont = np.zeros((padded_size,), dtype=np.int64)
    #     for i, u in enumerate(units):
    #         current_cmd_type[i] = u['current_cmd']['cmd_type']
    #         current_cmd_cont[i] = u['current_cmd_cont']

    #     return current_cmd_type, current_cmd_cont

    def _process_count(self, entry, *, enemy_only=False):
        visibility_offset = 0
        army_offset = visibility_offset + len(gc.Visibility)
        enemy_offset = army_offset + len(gc.UnitTypes) - 1
        resource_offset = enemy_offset + len(gc.UnitTypes) - 1
        num_channels = resource_offset + 1
        assert CoachDataset.get_num_count_channels() == num_channels

        counts = np.zeros((num_channels, ), dtype=np.float32)

        # fill in visibility
        visibility = np.array(entry['map']['visibility'])
        visible = (visibility == gc.Visibility.VISIBLE.value).sum()
        seen = (visibility == gc.Visibility.SEEN.value).sum()
        invisible = (visibility == gc.Visibility.INVISIBLE.value).sum()

        counts[0] = visible / visibility.size
        counts[1] = seen / visibility.size
        counts[2] = invisible / visibility.size
        assert(counts[0] + counts[1] + counts[2] == 1)

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

                counts[type_offset] += 1

        if enemy_only:
            counts = counts[enemy_offset : resource_offset]

        return counts

    def _process_instruction(self, ins):
        assert self.inst_dict is not None

        instruction, length = self.inst_dict.parse(ins, True)
        return np.array(instruction, dtype=np.int64), length


if __name__ == '__main__':
    import pickle

    inst_dict = pickle.load(open(
        '/private/home/hengyuan/scratch/rts_data/dataset_ref2/dict.pt', 'rb'))
    inst_dict.set_max_sentence_length(20)

    dataset = CoachDataset(
        '/private/home/hengyuan/scratch/rts_data/dataset3_nofow/dev.json',
        # '/private/home/hengyuan/scratch/rts_data/dataset_ref2/dev.json',
        0.98,
        11,
        50,
        25,
        inst_dict,
        20,
    )
    # compute_cache(dataset, 0)
