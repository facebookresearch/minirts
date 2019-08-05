# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import pprint
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# import matplotlib.pyplot
# matplotlib.pyplot.switch_backend('agg')
# import matplotlib.pyplot as plt


def convert_to_raw_instruction(inst, max_raw_chars):
    raw = list(inst.encode())
    if len(raw) > max_raw_chars:
        return raw[ : max_raw_chars]
    raw = raw  + ([-1] * (max_raw_chars - len(raw)))
    return raw


def parse_dataset(replay_paths):
    replays = []
    for replay_path in tqdm(replay_paths):
        replay = Replay(replay_path)
        replays.append(replay)
    return replays


def analyze_dataset(dataset):
    target_cmd_types = {cmd.name: 0 for cmd in global_consts.CmdTypes}
    cont_target_cmd_types = {cmd.name: 0 for cmd in global_consts.CmdTypes}

    for entry in dataset.data:
        if int(entry['tick'][4:]) >= 300:
            continue

        print(entry['instruction'])
        for my_unit in entry['my_units']:
            if my_unit['unit_type'] != 1:
                continue
            current_cmd = my_unit['current_cmd']
            target_cmd = my_unit['target_cmd']
            target_type = global_consts.CmdTypes(target_cmd['cmd_type']).name
            current_type = global_consts.CmdTypes(current_cmd['cmd_type']).name
            if target_type == 'CONT':
            # if target_cmd == my_unit['current_cmd']:
                cont_target_cmd_types[current_type] += 1
            else:
                target_cmd_types[target_type] += 1

    total = np.sum(list(target_cmd_types.values()))
    cont_total = np.sum(list(cont_target_cmd_types.values()))

    print('Tran---: %.2f' % (100 * total / (total + cont_total)))
    for k, v in target_cmd_types.items():
        print('{0: <35}: {1:10d}, {2:.2f}'.format(k, v, 100 * (v / total)))

    print('Cont---: %.2f' % (100 * cont_total / (total + cont_total)))
    for k, v in cont_target_cmd_types.items():
        print('{0: <35}: {1:10d}, {2:.2f}'.format(k, v, 100 * (v / cont_total)))


def analyze_build(dataset, cmd_type):
    def _check_new_construction(current, previous, build_cmd):
        build_type = build_cmd['target_type']

        # print('check building:', global_consts.UnitTypes(build_type).name)

        for my_unit in current['my_units']:
            unit_type = my_unit['unit_type']
            if unit_type != build_type:
                continue

            new_unit = True
            unit_id = my_unit['unit_id']
            for prev_unit in previous['my_units']:
                if unit_id == prev_unit['unit_id']:
                    new_unit = False
            if new_unit:
                return True

        return False

    def _analyze(entrys, cmd_type):
        build_building_count = 0
        target_cont_count = 0

        targets = defaultdict(int)
        finished_targets = defaultdict(int)

        for i, entry in enumerate(entrys):
            for my_unit in entry['my_units']:
                current_cmd = my_unit['current_cmd']
                target_cmd = my_unit['target_cmd']

                current_type = global_consts.CmdTypes(current_cmd['cmd_type']).name
                target_type = global_consts.CmdTypes(target_cmd['cmd_type']).name

                if current_type == cmd_type:
                    build_building_count += 1
                    if target_type == 'CONT':
                        target_cont_count += 1
                    else:
                        # cmd changes
                        targets[target_type] += 1
                        if i == len(entrys) - 1:
                            continue
                        # print(i)
                        if _check_new_construction(entrys[i+1], entrys[i], current_cmd):
                            finished_targets[target_type] += 1

        print('%s: %d, target cont: %d, percent %.2f' % (
            cmd_type,
            build_building_count,
            target_cont_count,
            target_cont_count / build_building_count
        ))

        for key in targets:
            total = targets[key]
            finished = finished_targets[key]
            print('target: %s, total: %d, building finished: %d, %.2f' % (
                key, total, finished, finished / total * 100))
        # pprint.pprint(targets)
        # pprint.pprint(finished_targets)

    _analyze(dataset.data, cmd_type)

    entrys = []
    for entry in dataset.data:
        all_cont = True
        for my_unit in entry['my_units']:
            target_cmd = my_unit['target_cmd']
            target_type = global_consts.CmdTypes(target_cmd['cmd_type']).name

            if target_type != 'CONT':
                all_cont = False

        if not all_cont:
            # pprint.pprint(entry)
            # break
            entrys.append(entry)
    print('%d entrys filtered' % (len(dataset.data) - len(entrys)))

    _analyze(entrys, cmd_type)


def check_actual_build_percent(entrys):
    i = 0
    num_builds = 0
    actual_num_builds = 0

    while i < len(entrys):
        instruction = entrys[i]['instruction']
        j = i
        check = ('build' in instruction)

        if check:
            print('checking for instruction:', instruction)
            num_builds += 1

        while j < len(entrys) and entrys[j]['instruction'] == instruction:
            if not check:
                j += 1
                continue

            for unit in entrys[j]['my_units']:
                target_cmd_type = unit['target_cmd']['cmd_type']
                # print(target_cmd_type)
                if target_cmd_type == gc.CmdTypes.BUILD_BUILDING.value \
                   or target_cmd_type == gc.CmdTypes.BUILD_UNIT.value :
                    build_type = unit['target_cmd']['target_type']
                    build_type = gc.UnitTypes(build_type).name.lower()
                    if build_type == 'guard_tower':
                        build_type = 'tower'
                    if build_type == 'town_hall':
                        build_type = 'hall'
                    # print(build_type)
                    if build_type in instruction:
                        print('found:', build_type)
                        actual_num_builds += 1
                        check = False

            j += 1

        i = j
    print('total: %d, actual: %d, percent: %.2f' %
          (num_builds, actual_num_builds, actual_num_builds / num_builds))


def write_tensor_to_image(tensor, path, size=(300, 300)):
    channel_names = [
        'x', 'y', 'visible', 'seen', 'invisible',
        'soil', 'sand', 'grass', 'rock', 'water', 'fog',
        'peasant', 'spearman', 'swordman', 'cavalry', 'dragon',
        'archer', 'catapult', 'barrack', 'blacksmith', 'stable',
        'workshop', 'aviary', 'archery', 'guard-tower', 'town-hall',
        'e-peasant', 'e-spearman', 'e-swordman', 'e-cavalry',
        'e-dragon', 'e-archer', 'e-catapult', 'e-barrack',
        'e-blacksmith', 'e-stable', 'e-workshop', 'e-guard-tower',
        'e-aviary', 'e-archery', 'e-town-hall',
        'resource'
    ]
    rows = 6
    cols = 7

    num_channels = tensor.shape[0]
    assert(num_channels == len(channel_names))
    fig, ax = plt.subplots(rows, cols, figsize=(cols*10, rows*10))

    for i in range(len(channel_names)):
        r = i // cols
        c = i % cols
        data = tensor[i]

        ax[r, c].imshow(data, cmap=matplotlib.cm.Greys_r, vmin=0, vmax=1)
        ax[r, c].set_title('c%d_%s' % (i, channel_names[i]), fontsize=50)
        ax[r, c].axis('off')

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


_correct2wrongs = {
    'attack': ['aattack', 'atack', 'atatck', 'attaack', 'attac', 'attacck', 'attach',
               'attacking', 'attacks', 'attak', 'attakc', 'attck', 'attk', 'atttack', ],
    'act': ['ack', ],
    'dragon': ['adragon', ],
    'allow': ['allows', ],
    'all peasant': ['allpeasants', ],
    'alright': ['alrighty',],
    'alternate': ['alternating'],
    'another': ['anotehr', 'anther', 'antoher'],
    'appear': ['appears', ],
    'approach': ['approaches', 'approaching'],
    'archer': ['archer(s)', 'archers', 'archrs', 'arhcer', 'arhcers'],
    'area': ['areas', ],
    'are not': ["aren't", ],
    'a tower': ['atower', ],
    'attacker': ['attackers', ],
    'attack the': ['attackthe', ],
    'avoid': ['avoiding', ],
    'barrack': ['barack', 'baracks', 'barracks', ],
    'base': ['bases', ],
    'build': ['biuld', 'buid', 'buil', 'builda', 'builds', 'builld', 'buld', 'bulid', ],
    'body': ['bodies', ],
    'bombard': ['bombarding', ],
    'builder': ['builders', ],
    'building': ['buildings', ],
    'cavalry': ['calavary', 'calavry', 'calvalry', 'calvaries',
                'calvary', 'calvarymen', 'calvery', 'cav', 'cavalary',
                'cavaliers', 'cavalries', 'cavalrys', 'cavarly',
                'cavary', 'cavlary', 'cavlry', 'cavs', 'chivalries',
                'horde', 'horse', 'horseman', 'horsemen', 'horses',
                'horsey', 'horseys', 'hroses', 'knight', 'knights', ],
    'can not': ["can't", 'cannot' 'cant'],
    'catapult': ['cata', 'catapalt', 'catapault', 'catapaults',
                 'catapuls', 'catapult', 'catapults', 'catas',
                 'catpult'], 'center': ['central', 'centre',],
    'chase': ['chasing', ],
    'circle': ['circling', ],
    'click': ['clicking', ],
    'close': ['closely', 'closes'],
    'cluster': ['clustered', 'clusters', ],
    'collect': ['collecting', ],
    'come': ['comes', 'coming'],
    'complete': ['completed',],
    'corner': ['corners', ],
    'create': ['crate', 'creat', 'created', 'creating', ],
    'crystal': ['crystals', ],
    'current': ['currently', ],
    'damage': ['damaged', ],
    'defend': ['defending', ],
    'deposit': ['deposits', ],
    'destroy': ['destory', 'destroyed', 'destroying', ],
    'do not': ["didn't", 'didnt', "doesn't", 'doesnt', "don't", 'dont'],
    'die': ['died', 'dies', ],
    'dragon': ['dragon;', 'dragons', 'drgaon', ],
    'duty': ['duties', ],
    'each other': ['eachother', ],
    'enemy': ['enemies', "enemy's", 'enemys', 'enermy', 'enim', 'enmey', 'enmy', ],
    'enemy base': ['enemybase', ],
    'peasant': ['epasant', 'epasants', 'epeasants', 'peasants', ],
    'field': ['fields', ],
    'friend': ['friends', ],
    'gap': ['gaps',],
    'guard': ['guadr', 'guar', 'guards', ],
    'guard tower': ['guard_tower', 'guardtower', ],
    'half': ['hal'],
    'hall': ['hall*', 'halls',],
    'harass': ['harrass', ],
    'keep': ['keeep', ],
    'kill': ['killhim', 'killl', ],
    'lake': ['lakes', ],
    'let us': ["let's", 'lets', ],
    'location': ['locations', ],
    'make': ['makea', 'makes', 'making'],
    'material': ['materials', ],
    'mineral': ['mienrals', 'mineals', 'minearls', 'minerals', 'minerla', 'minerlas', ],
    'mine': ['miner', 'miners', 'mines', 'ming', 'minig', 'mining'],
    'mobilize': ['mobilizing',],
    'move': ['moves', 'moving'],
    'near': ['neard',],
    'need': ['needs', 'needed'],
    'node': ['nodes', ],
    'north': ['norht', ],
    'northeast': ['north-east', ],
    'northwest': ['north-west', ]
}
