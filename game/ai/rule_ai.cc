// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#include "ai/rule_ai.h"
#include "ai/comm_ai.h"
#include "engine/utils.h"
#include "engine/mc_strategy_actor.h"

#include <vector>

bool RandomizedSingleUnitTypeAI::act(const RTSStateExtend& state, RTSAction* action) {
  action->Init(getId(), GameDef::GetNumAction(), RTSAction::RULE_BASED);

  if (inactiveUntil_ != -1 && inactiveUntil_ < state.GetTick()) {
    return true;
  }

  const auto& env = state.env();
  const auto& receiver = state.receiver();
  MCStrategyActor strategy_actor(receiver, getId());

  // fail to gather infomation, no need to act
  if (!strategy_actor.GatherInfo(env, buildQueue, respectFow())) {
    return false;
  }

  std::vector<int64_t> action_state(GameDef::GetNumAction(), 0);
  strategy_actor.Prepare(&action_state);

  const auto& preload = strategy_actor.preload();
  int num_tower = preload.NumUnit(GUARD_TOWER, true);
  if (buildTower_ && num_tower < 2) {
    strategy_actor.BuildTower(&action_state);
  }
  strategy_actor.BuildScoutAttack(
      env.GetGameDef(), unitType_, &action_state, minArmy_, maxArmy_);
  strategy_actor.RushAgainstDragon(&action_state);

  action->SetAction(std::move(action_state));
  return strategy_actor.ActByState(env, this, action);
}

bool StrongAI::act(const RTSStateExtend& state, RTSAction* action) {
  action->Init(getId(), GameDef::GetNumAction(), RTSAction::RULE_BASED);

  const auto& env = state.env();
  const auto& receiver = state.receiver();
  MCStrategyActor strategy_actor(receiver, getId());

  // fail to gather infomation, no need to act
  if (!strategy_actor.GatherInfo(env, buildQueue, respectFow())) {
    return false;
  }

  const auto& preload = strategy_actor.preload();
  const auto& enemy_troops = preload.EnemyTroops();

  std::vector<int64_t> action_state(GameDef::GetNumAction(), 0);
  strategy_actor.Prepare(&action_state);

  // decide which strategy to use
  UnitType target_type = INVALID_UNITTYPE;

  const std::vector<UnitType> army_search_order = {
      DRAGON, SWORDMAN, SPEARMAN, CAVALRY, ARCHER};
  const std::vector<UnitType> factory_search_order = {
      WORKSHOP, BLACKSMITH, BARRACK, STABLE};

  // update lastSeenEnemyUnit
  for (auto ut : army_search_order) {
    if (enemy_troops[ut].size() > 0) {
      lastSeenEnemyUnit_ = ut;
      // std::cout << "lastSeenEnemyUnit: " << ut
      //           << ", count: " << enemy_troops[ut].size() << std::endl;
      break;
    }
  }

  if (lastSeenEnemyUnit_ != INVALID_UNITTYPE) {
    target_type = lastSeenEnemyUnit_;
  } else {
    for (auto ut : factory_search_order) {
      if (enemy_troops[ut].size() > 0) {
        target_type = ut;
        break;
      }
    }
  }

  if (preload.EnemyBaseTargets().size()|| enemy_troops[GUARD_TOWER].size()) {
    scouted_ = true;
  }

  if (target_type != INVALID_UNITTYPE) {
    if (target_type == ARCHER) {
      if (preload.NumUnit(BLACKSMITH, true) > 0) {
        unitType_ = SWORDMAN;
      } else if (preload.NumUnit(STABLE, true) > 0) {
        unitType_ = CAVALRY;
      } else if (preload.NumUnit(BARRACK, true) > 0) {
        unitType_ = SPEARMAN;
      } else {
        std::uniform_int_distribution<int> dist(0, 2);
        int rand = dist(rng);
        if (rand == 0) {
          unitType_ = SPEARMAN;
        } else if (rand == 1) {
          unitType_ = SWORDMAN;
        } else {
          unitType_ = CAVALRY;
        }
      }
    } else {
      auto it = UnitCounteractMap.find(target_type);
      if (it != UnitCounteractMap.end()) {
        unitType_ = it->second;
      } else {
        unitType_ = DRAGON;
      }
    }
    strategy_actor.BuildScoutAttack(
        env.GetGameDef(), unitType_, &action_state, minArmy_, maxArmy_);
  } else {
    if (!scouted_) {
      strategy_actor.ConservativeScout(&action_state);
    }

    if (preload.NumUnit(GUARD_TOWER, true) < 2) {
      strategy_actor.BuildTower(&action_state);
    }

    if (preload.Resource() >= 250) {
      strategy_actor.BuildScoutAttack(
          env.GetGameDef(), DRAGON, &action_state, minArmy_, maxArmy_);
    }
  }
  strategy_actor.RushAgainstDragon(&action_state);

  action->SetAction(std::move(action_state));
  return strategy_actor.ActByState(env, this, action);
}
