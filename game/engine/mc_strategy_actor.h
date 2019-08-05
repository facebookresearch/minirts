// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include "engine/mc_rule_actor.h"

custom_enum(MINI_STRATEGY,
            STRATEGY_IDLE,
            // STRATEGY_IDLE_BUILD,
            STRATEGY_BUILD_SWORDMAN,
            STRATEGY_BUILD_SPEARMAN,
            STRATEGY_BUILD_CAVALRY,
            STRATEGY_BUILD_ARCHER,
            STRATEGY_BUILD_DRAGON,
            STRATEGY_BUILD_CATAPULT,
            STRATEGY_BUILD_TOWER,
            STRATEGY_BUILD_PEASANT,
            // STRATEGY_IDLE_SCOUT,
            STRATEGY_SCOUT,
            // STRATEGY_IDLE_ATTACK_BASE,
            STRATEGY_ATTACK_BASE,
            NUM_STRATEGY);

static const std::map<UnitType, UnitType> UnitCounteractMap = {
    {DRAGON, ARCHER},
    {SWORDMAN, CAVALRY},
    {BLACKSMITH, CAVALRY},
    {SPEARMAN, SWORDMAN},
    {BARRACK, SWORDMAN},
    {CAVALRY, SPEARMAN},
    {STABLE, SPEARMAN},
    {WORKSHOP, DRAGON}
    // {TOWER, CATAPULT} TODO: think about this
};

class MCStrategyActor : public MCRuleActor {
 public:
  MCStrategyActor(const CmdReceiver& receiver, PlayerId playerId)
      : MCRuleActor(receiver, playerId) {
  }

  MCStrategyActor(const CmdReceiver& receiver,
                  const Preload& preload,
                  PlayerId player_id)
      : MCRuleActor(receiver, preload, player_id) {
  }

  // thses functions convert strategy_vector to action_state vector
  void BuildScoutAttack(const GameDef& gamedef,
                        UnitType unitType,
                        std::vector<int64_t>* action,
                        int minArmy,
                        int maxArmy);

  void BuildScoutAttackTwoUnits(const GameDef& gamedef,
                                UnitType unitType1,
                                UnitType unitType2,
                                std::vector<int64_t>* action,
                                int minArmy1,
                                int minArmy2,
                                int maxArmy);

  void RushAgainstDragon(std::vector<int64_t>* action);

  void BuildArmy(const GameDef& gamedef,
                 UnitType unitType,
                 std::vector<int64_t>* action);

  void AttackBase(std::vector<int64_t>* action);

  void Expand(std::vector<int64_t>* action);

  void BuildTower(std::vector<int64_t>* action);

  void BuildPeasant(std::vector<int64_t>* action);

  void ConservativeScout(std::vector<int64_t>* action);

  void Scout(std::vector<int64_t>* action);

  // set gather, defend, and build peasant if needed
  // used by rule ai
  void Prepare(std::vector<int64_t>* action);

  // convert strategy vector to cmds (both stored inside action)
  // should only be used by trained AI
  bool ActByStrategy(const GameEnv& env, AI* ai, RTSAction* action);

  void SetNumPeasant(int n) {
    optimalNumPeasant_ = n;
  }

 private:
  void BuildPeasantIfNeeded(int min_num_peasant, std::vector<int64_t>* action);

  bool DefendIfNeeded(std::vector<int64_t>* action);

  bool needDefend_;

  int optimalNumPeasant_ = 5;
};
