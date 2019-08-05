# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import json
import common_utils.global_consts as gc


def remove_duplicated_targets(frames, verbose=False):
    num_target_removed = 0
    total_num_target = 0

    for entry in frames:
        if entry is None or type(entry) == float:
            continue
        if 'targets' not in entry:
            continue

        new_targets = {}
        for target in entry['targets']:
            unit_id = target['unit_id']
            if unit_id not in new_targets:
                new_targets[unit_id] = target
            else:
                old_tick = int(new_targets[unit_id]['tick'])
                new_tick = int(target['tick'])
                # print('remove: old tick:', old_tick, ' new tick:', new_tick)
                if new_tick > old_tick:
                    new_targets[unit_id] = target

        total_num_target += len(entry['targets'])
        num_target_removed += len(entry['targets']) - len(new_targets)
        entry['targets'] = list(new_targets.values())

    if verbose:
        print('remove duplicated: %s out of %s'
              % (num_target_removed, total_num_target))
    return frames


def unit_has_target_cmd(unit_id, targets):
    found = 0
    idx = -1
    for i, t in enumerate(targets):
        if t['unit_id'] == unit_id:
            assert(found == 0)
            found = 1
            idx = i
    if found == 1:
        return targets[idx]
    else:
        return None


def same_cmd(cur, target):
    fields = ['cmd_type',
              'target_attack_idx',
              'target_gather_idx',
              'target_id',
              'target_type',
              'target_x',
              'target_y']

    for field in fields:
        if cur[field] != target[field]:
            return False
    return True


def convert_to_cont(cmd):
    cmd['cmd_type'] = gc.CmdTypes.CONT.value
    cmd['target_attack_idx'] = 0
    cmd['target_gather_idx'] = 0
    cmd['target_id'] = 0
    cmd['target_type'] = 0
    cmd['target_x'] = 0
    cmd['target_y'] = 0
    return cmd


def convert_to_idle(cmd):
    cmd['cmd_type'] = gc.CmdTypes.IDLE.value
    cmd['target_attack_idx'] = 0
    cmd['target_gather_idx'] = 0
    cmd['target_id'] = 0
    cmd['target_type'] = 0
    cmd['target_x'] = 0
    cmd['target_y'] = 0
    return cmd


def process_entry(entry, prefix):
    entry['unique_id'] = '%s-%s' % (prefix, entry['tick'])

    # attach target to my unit
    my_units = []
    for unit in entry['my_units']:
        unit_id = unit['unit_id']

        target_cmd = None
        if 'targets' in entry:
            target_cmd = unit_has_target_cmd(unit_id, entry['targets'])

        if target_cmd is None:
            target_cmd = convert_to_cont(unit['current_cmd'].copy())
        else:
            assert target_cmd['cmd_type'] >= 0
            assert target_cmd['target_id'] >= 0
            assert target_cmd['target_type'] >= 0
            for key in ['target_x', 'target_y']:
                if target_cmd[key] < 0:
                    target_cmd[key] = 0
                if target_cmd[key] >= 31:
                    target_cmd[key] = 31
            if int(entry['tick'][4:]) > int(target_cmd['tick']):
                print('warning, something is wrong:',
                      int(entry['tick'][4:]), int(target_cmd['tick']))

            # assert int(entry['tick'][4:]) <= int(target_cmd['tick'])

            if same_cmd(target_cmd, unit['current_cmd']):
                # 1) idle is fixed
                # 2) gather convert to continue
                # 3) attack is fixed
                # 4) build building do nothing
                # 5) build unit do nothing
                # 6) move do nothing
                if target_cmd['cmd_type'] == gc.CmdTypes.GATHER.value:
                    target_cmd = convert_to_cont(target_cmd)

        unit['target_cmd'] = target_cmd

        # "thrown-forward" sections are thrown away
        assert target_cmd['target_gather_idx'] != -1
        assert target_cmd['target_attack_idx'] != -1

        my_units.append(unit)

    entry['my_units'] = my_units

    # remove targets
    if 'targets' in entry:
        entry.pop('targets')

    return entry


def reassign_index(entry):
    army_id2idx = {}
    enemy_id2idx = {}
    resource_id2idx = {}

    for key, table in [
            ('my_units', army_id2idx),
            ('enemy_units', enemy_id2idx),
            ('resource_units', resource_id2idx)]:
        for unit in entry[key]:
            unit_id = unit['unit_id']
            assert unit_id not in table, ('duplicated unit id %s, %s' % (unit_id, table))
            table[unit_id] = len(table)

    for unit in entry['my_units']:
        unit_id = unit['unit_id']
        unit['idx'] = army_id2idx[unit_id]

        cmd_type = unit['current_cmd']['cmd_type']
        target_id = unit['current_cmd']['target_id']

        if cmd_type == gc.CmdTypes.ATTACK.value:
            # print('searching attack ', target_id, 'in :' )
            # print(enemy_id2idx)
            if target_id in enemy_id2idx:
                unit['current_cmd']['target_attack_idx'] = enemy_id2idx[target_id]
            else:
                unit['current_cmd'] = convert_to_idle(unit['current_cmd'])
        elif cmd_type == gc.CmdTypes.GATHER.value:
            # print('searching gather ', target_id, 'in :' )
            # print(resource_id2idx)
            if target_id in resource_id2idx:
                unit['current_cmd']['target_gather_idx'] = resource_id2idx[target_id]
            else:
                unit['current_cmd'] = convert_to_idle(unit['current_cmd'])

    # process cmds
    if 'targets' not in entry:
        return entry

    filtered_targets = []
    for target in entry['targets']:
        cmd_type = target['cmd_type']
        target_id = target['target_id']
        keep = True
        if cmd_type == gc.CmdTypes.ATTACK.value:
            if target_id in enemy_id2idx:
                target['target_attack_idx'] = enemy_id2idx[target_id]
            else:
                keep = False
        elif cmd_type == gc.CmdTypes.GATHER.value:
            if target_id in resource_id2idx:
                target['target_gather_idx'] = resource_id2idx[target_id]
            else:
                keep = False
        if keep:
            filtered_targets.append(target)
    entry['targets'] = filtered_targets
    return entry


def process_game(game, prefix):
    """
    process a json object of a game,
    return list of data points in training format
    """
    data = []
    # with open(state_file, 'r') as fin:
    #     js = json.load(fin)
    # print('processing %s' % state_file)
    game = remove_duplicated_targets(game)

    for entry in game:
        if entry is None or type(entry) == float or not 'tick' in entry:
            continue
        # # there can be one empty entry for human dataset
        # if:
        #     continue

        # print('reassigning index')
        entry = reassign_index(entry)
        entry = process_entry(entry, prefix)
        data.append(entry)

    return data
