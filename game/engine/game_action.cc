// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#include "engine/game_action.h"
#include "engine/mc_rule_actor.h"

bool RTSAction::Send(const GameEnv& env, CmdReceiver& receiver) {
  // TODO: temp fix, should move preload computation into action
  if (_type == ActionType::CMD_BASED) {
    MCRuleActor rule_actor(receiver, _player_id);
    // const Player& player = env.GetPlayer(_player_id);

    rule_actor.GatherInfo(env, std::list<UnitType>(), true);
    rule_actor.SetTowerAutoAttack(env, &_cmds);
    // rule_actor.SetAutoDefense(env, player, &_cmds);
  }
  return true;
}
