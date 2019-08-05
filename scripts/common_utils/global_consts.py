# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# define some globals, ideally should be auto-synced with c++
from enum import Enum
import numpy as np

good_insts = [
    0, # attack
    1, # send all peasant to mine
    2, # build a workshop
    8, # build a stable
    11, # build a dragon
    15, # build a guard tower
    16, # build a catapult
    27, # build a blacksmith
    41, # build peasant
    46, # build a barrack
    56, # build arhcer
    66, # build cavalry
    195, # build a spearman
    331, # build another swordman
    239, # keep scouting
    265, # send one peasant to scout
]
good_inst_mask = np.zeros(500)
good_inst_mask[good_insts] = 1


better_insts = [
0, # attack
1, # send all peasant to mine
2, # build a workshop
3, # retreat
4, # attack peasant
6, # send idle peasant to mine
8, # build a stable
11, # build a dragon
15, # build a guard tower
16, # build a catapult
20, # build another peasant
22, # build 4 peasant
26, # attack dragon
30, # build 2 archer
33, # build 3 more peasant
34, # back to mining
50, # mine with all peasant
51, # attack town hall
66, # build cavalry
121, # attack tower
152, # build 3 spearman
233, # scout for resources
249, # attack peasant with dragon
252, # attack tower with catapult
387, # make another peasant to mine
]
better_inst_mask = np.zeros(500)
better_inst_mask[better_insts] = 1


class CmdTypes(Enum):
    # IDLE means doing_nothing as input, means do_nothing (continue) as output
    IDLE = 0
    GATHER = 1
    ATTACK = 2
    BUILD_BUILDING = 3
    BUILD_UNIT = 4
    MOVE = 5
    CONT = 6


class UnitTypes(Enum):
    RESOURCE = 0
    PEASANT = 1
    SPEARMAN = 2
    SWORDMAN = 3
    CAVALRY = 4
    DRAGON = 5
    ARCHER = 6
    CATAPULT = 7
    BARRACK = 8
    BLACKSMITH = 9
    STABLE = 10
    WORKSHOP = 11
    AVIARY = 12
    ARCHERY = 13
    GUARD_TOWER = 14
    TOWN_HALL = 15

    @classmethod
    def get_id(cls, name):
        assert name in UNIT_TYPE_TO_IDX, 'invalid unit type: %s' % name
        return UNIT_TYPE_TO_IDX[name]

UNIT_TYPE_TO_IDX = {e.name : e.value for e in UnitTypes}


class Visibility(Enum):
    VISIBLE = 0
    SEEN = 1
    INVISIBLE = 2


class Terrain(Enum):
    SOIL = 0
    SAND = 1
    GRASS = 2
    ROCK = 3
    WATER = 4
    FOG = 5


class RuleAction(Enum):
    IDLE = 0,
    BUILD_PEASANT = 1
    BUILD_SWORDMAN = 2
    BUILD_SPEARMAN = 3
    BUILD_CAVALRY = 4
    BUILD_ARCHER = 5
    BUILD_DRAGON = 6
    BUILD_CATAPULT = 7
    BUILD_BARRACK = 8
    BUILD_BLACKSMITH = 9
    BUILD_STABLE = 10
    BUILD_WORKSHOP = 11
    BUILD_GUARD_TOWER = 12
    BUILD_TOWN_HALL = 13
    ATTACK_BASE = 14
    ATTACK_IN_RANGE = 15
    ATTACK_TOWER_RUSH = 16
    ATTACK_PEASANT_BASE = 17
    DEFEND = 18
    SCOUT = 19
    GATHER = 20

NUM_RULE = len(RuleAction)  # size of action produced by rule ai


RuleStrategy = [
    'STRATEGY_IDLE',
    'STRATEGY_BUILD_SWORDMAN',
    'STRATEGY_BUILD_SPEARMAN',
    'STRATEGY_BUILD_CAVALRY',
    'STRATEGY_BUILD_ARCHER',
    'STRATEGY_BUILD_DRAGON',
    'STRATEGY_BUILD_CATAPULT',
    'STRATEGY_BUILD_TOWER',
    'STRATEGY_BUILD_PEASANT',
    'STRATEGY_SCOUT',
    'STRATEGY_ATTACK_BASE',
]

NUM_STRATEGY = len(RuleStrategy)


MAP_X = 32
MAP_Y = 32
