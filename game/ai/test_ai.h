// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once
#include <stdlib.h>
#include <limits>
#include <random>
#include <vector>

#include "ai/ai.h"
// #include "elf/utils/utils.h"
#include "game_MC/mc_strategy_actor.h"

class TestBuildAI : public AI {
 public:
  TestBuildAI(const AIOption& opt, int threadId) : AI(opt, threadId) {}

  int tick = 0;

  bool act(const RTSState& state, RTSAction* action) {
    action->Init(getID(), GameDef::GetNumAction(), RTSAction::RULE_BASED);

    const auto& env = state.env();
    const auto& receiver = state.receiver();
    MCStrategyActor strategy_actor(receiver, getID());

    // fail to gather infomation, no need to act
    if (!strategy_actor.GatherInfo(env, buildQueue, respectFow())) {
      return false;
    }

    std::vector<int64_t> action_state(GameDef::GetNumAction(), 0);
    action_state[STATE_GATHER] = 1;

    const auto& preload = strategy_actor.preload();

    if (tick == 0 && preload.NumUnit(GUARD_TOWER, true) < 1) {
      action_state[STATE_BUILD_GUARD_TOWER] = 1;
      tick += 1;
    } else if (tick == 1 && preload.NumUnit(STABLE, true) < 1) {
      action_state[STATE_BUILD_STABLE] = 1;
      tick += 1;
    } else if (tick == 2 && preload.NumUnit(WORKSHOP, true) < 1) {
      action_state[STATE_BUILD_WORKSHOP] = 1;
      tick += 1;
    }
    if (preload.NumUnit(BARRACK, true) < 1) {
      action_state[STATE_BUILD_BARRACK] = 1;
    }
    strategy_actor.ConservativeScout(&action_state);
    action->SetAction(std::move(action_state));
    strategy_actor.ActByState(env, this, action);
    return true;
  }
};

class TestAttackAI : public AI {
 public:
  TestAttackAI(const AIOption& opt, int threadId) : AI(opt, threadId) {}

  bool act(const RTSState& state, RTSAction* action) {
    action->Init(getID(), GameDef::GetNumAction(), RTSAction::RULE_BASED);

    const auto& env = state.env();
    const auto& receiver = state.receiver();
    MCStrategyActor strategy_actor(receiver, getID());

    // fail to gather infomation, no need to act
    if (!strategy_actor.GatherInfo(env, buildQueue, respectFow())) {
      return false;
    }

    std::vector<int64_t> action_state(GameDef::GetNumAction(), 0);
    action_state[STATE_GATHER] = 1;

    const auto& preload = strategy_actor.preload();

    if (preload.NumUnit(BARRACK, true) < 1) {
      action_state[STATE_BUILD_BARRACK] = 1;
    }
    if (preload.NumUnit(WORKSHOP, true) < 1) {
      action_state[STATE_BUILD_WORKSHOP] = 1;
    }

    // strategy_actor.ConservativeScout(&action_state);
    action_state[STATE_SCOUT] = 1;

    if (preload.NumUnit(SPEARMAN, true) >= 2 &&
        preload.NumUnit(CATAPULT, true) >= 2)  {
      action_state[STATE_ATTACK_IN_RANGE] = 1;
      action_state[STATE_ATTACK_BASE] = 1;
    } else if (preload.NumUnit(SPEARMAN, true) < 2) {
      action_state[STATE_BUILD_SPEARMAN] = 1;
    } else if (preload.NumUnit(CATAPULT, true) < 2) {
      action_state[STATE_BUILD_CATAPULT] = 1;
    }

    action_state[STATE_DEFEND] = 1;
    action->SetAction(std::move(action_state));
    strategy_actor.ActByState(env, this, action);
    return true;
  }
};

class TestBuildQueueAI : public AI {
 public:
  TestBuildQueueAI(const AIOption& opt, int threadId) : AI(opt, threadId) {}

  bool act(const RTSState& state, RTSAction* action) {
    action->Init(getID(), GameDef::GetNumAction(), RTSAction::RULE_BASED);

    const auto& env = state.env();
    const auto& receiver = state.receiver();
    MCRuleActor rule_actor(receiver, getID());

    // fail to gather infomation, no need to act
    if (!rule_actor.GatherInfo(env, buildQueue, respectFow())) {
      return false;
    }

    std::vector<int64_t> action_state(GameDef::GetNumAction(), 0);
    action_state[STATE_GATHER] = 1;

    const auto& preload = rule_actor.preload();

    if (preload.NumUnit(BARRACK, true) < 1) {
      action_state[STATE_BUILD_BARRACK] = 1;
    }
    if (preload.NumUnit(WORKSHOP, true) < 1) {
      action_state[STATE_BUILD_WORKSHOP] = 1;
    }
    if (preload.NumUnit(BLACKSMITH, true) < 1) {
      action_state[STATE_BUILD_BLACKSMITH] = 1;
    }
    if (preload.NumUnit(STABLE, true) < 1) {
      action_state[STATE_BUILD_STABLE] = 1;
    }

    action->SetAction(std::move(action_state));
    rule_actor.ActByState(env, this, action);
    return true;
  }
};

class TestVoidAI : public AI {
 public:
  TestVoidAI(const AIOption& opt, int threadId) : AI(opt, threadId) {}

  bool act(const RTSState&, RTSAction* action) {
    action->Init(getID(), GameDef::GetNumAction(), RTSAction::RULE_BASED);
    return true;
  }
};
