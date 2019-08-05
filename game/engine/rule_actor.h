// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include "cmd.h"
#include "cmd_interface.h"
#include "cmd_specific.gen.h"
#include "game_env.h"
#include "preload.h"
#include <algorithm>
#include <sstream>
#include <stack>

custom_enum(AIState,
            STATE_IDLE = 0,
            STATE_BUILD_PEASANT,
            STATE_BUILD_SWORDMAN,
            STATE_BUILD_SPEARMAN,
            STATE_BUILD_CAVALRY,
            STATE_BUILD_ARCHER,
            STATE_BUILD_DRAGON,
            STATE_BUILD_CATAPULT,
            STATE_BUILD_BARRACK,
            STATE_BUILD_BLACKSMITH,
            STATE_BUILD_STABLE,
            STATE_BUILD_WORKSHOP,
            STATE_BUILD_GUARD_TOWER,
            STATE_BUILD_TOWN_HALL,
            STATE_ATTACK_BASE,
            STATE_ATTACK_IN_RANGE,
            STATE_ATTACK_TOWER_RUSH,
            STATE_ATTACK_PEASANT_BASE,
            STATE_DEFEND,
            STATE_SCOUT,
            STATE_GATHER,
            NUM_AISTATE);

// custom_enum(
//     FlagState,
//     FLAGSTATE_START = 0,
//     FLAGSTATE_GET_FLAG,
//     FLAGSTATE_ATTACK_FLAG,
//     FLAGSTATE_ESCORT_FLAG,
//     FLAGSTATE_PROTECT_FLAG, // FLAGSTATE_ATTACK, FLAGSTATE_MOVE,
//     NUM_FLAGSTATE);

// Some easy macros
#define _M(...) CmdBPtr(new CmdMove(INVALID, __VA_ARGS__))
#define _A(target) CmdBPtr(new CmdAttack(INVALID, target))
#define _G(...) CmdBPtr(new CmdGather(INVALID, __VA_ARGS__))
#define _B(...) CmdBPtr(new CmdBuild(INVALID, __VA_ARGS__))

// // Region commands.
// // BUILD_PEASANT: for all idle town_halls in this region, build a worker.
// // BUILD_BARRACK: Pick an idle/gathering worker in this region and
// // build a barrack.
// // BUILD_MELEE_TROOP: For all barracks in this region, build melee
// // troops.
// // BUILD_RANGE_TROOP: For all barracks in this region, build range
// // troops.
// // ATTACK: For all troops (except for workers) in this region, attack
// // the opponent town_hall.
// // ATTACK_IN_RANGE: For all troops (including workers) in this region,
// // attack enemy in range.
// custom_enum(
//     AIStateRegion,
//     SR_NOCHANGE = 0,
//     SR_BUILD_PEASANT,
//     SR_BUILD_BARRACK,
//     SR_BUILD_MELEE_TROOP,
//     SR_BUILD_RANGE_TROOP,
//     SR_ATTACK,
//     SR_ATTACK_IN_RANGE,
//     NUM_SR);

// Information of the game used for AI decision.
class RuleActor {
 public:
  RuleActor(const CmdReceiver& receiver, PlayerId player_id)
      : _receiver(receiver)
      , _copied_preload(false)
      , _player_id(player_id) {
  }

  RuleActor(const CmdReceiver& receiver,
            const Preload& preload,
            PlayerId player_id)
      : _receiver(receiver)
      , _preload(preload)
      , _copied_preload(true)
      , _player_id(player_id) {
  }

  const Preload& preload() const {
    return _preload;
  }

  Preload& preload() {
    return _preload;
  }

  bool GatherInfo(const GameEnv& env,
                  const std::list<UnitType>& build_queue,
                  bool respect_fow);

 protected:
  bool store_cmd(const Unit*, CmdBPtr&& cmd, AssignedCmds* m) const;

  // void batch_store_cmds(
  //     const std::vector<const Unit*>& subset,
  //     const CmdBPtr& cmd,
  //     bool preemptive,
  //     AssignedCmds* m) const;

  const CmdReceiver& _receiver;
  Preload _preload;
  bool _copied_preload;
  PlayerId _player_id;
};
