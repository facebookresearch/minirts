// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include "ai/ai.h"
#include "engine/rule_actor.h"
#include <list>

static const std::map<UnitType, AIState> StateBuildMap = {
    {PEASANT, STATE_BUILD_PEASANT},
    {SWORDMAN, STATE_BUILD_SWORDMAN},
    {SPEARMAN, STATE_BUILD_SPEARMAN},
    {CAVALRY, STATE_BUILD_CAVALRY},
    {ARCHER, STATE_BUILD_ARCHER},
    {DRAGON, STATE_BUILD_DRAGON},
    {CATAPULT, STATE_BUILD_CATAPULT},
    {BARRACK, STATE_BUILD_BARRACK},
    {STABLE, STATE_BUILD_STABLE},
    {WORKSHOP, STATE_BUILD_WORKSHOP},
    {BLACKSMITH, STATE_BUILD_BLACKSMITH},
    {GUARD_TOWER, STATE_BUILD_GUARD_TOWER},
    {TOWN_HALL, STATE_BUILD_TOWN_HALL},
};

class MCRuleActor : public RuleActor {
 public:
  MCRuleActor(const CmdReceiver& receiver, PlayerId player_id)
      : RuleActor(receiver, player_id) {
  }

  MCRuleActor(const CmdReceiver& receiver,
              const Preload& preload,
              PlayerId player_id)
      : RuleActor(receiver, preload, player_id) {
  }

  // Act by a state array, used by MiniRTS
  bool ActByState(const GameEnv& env, AI* ai, RTSAction* action);

  bool SetTowerAutoAttack(const GameEnv& env, AssignedCmds* assigned_cmds);

  bool PeasantDefend(const GameEnv& env, AssignedCmds* assigned_cmds);

  // bool SetAutoDefense(
  //     const GameEnv& env, const Player& player, AssignedCmds* assigned_cmds);

 private:
  void build_units(const GameEnv& env,
                   const TownHall& town_hall,
                   const std::vector<int64_t>& action_state,
                   const float building_radius_check,
                   size_t max_build_queue_size,
                   std::list<UnitType>* build_queue,
                   std::mt19937* rng,
                   AssignedCmds* assigned_cmds);

  bool build_unit(UnitType unit_type,
                  const GameEnv& env,
                  const TownHall& focus_town_hall,
                  float building_l1_radius,
                  AssignedCmds* assigned_cmds);

  bool build_town_hall(const GameEnv& env,
                       float building_radius_check,
                       AssignedCmds* assigned_cmds);

  bool gather(float resource_scale, AssignedCmds* assigned_cmds);

  bool attack(const GameDef& gamedef,
              const std::vector<const Unit*>& my_units,
              const std::vector<const Unit*>& enemy_units,
              float d_sqr,
              AssignedCmds* assigned_cmds);

  bool attack_base(const GameDef& gamedef, AssignedCmds* assigned_cmds);

  bool attack_peasant_base(const GameDef& gamedef, AssignedCmds* assigned_cmds);

  bool attack_tower_rush(const GameEnv& env,
                         AssignedCmds* assigned_cmds,
                         const float building_l1_radius);

  bool attack_in_range(const GameDef& gamedef, AssignedCmds* assigned_cmds);

  bool defend(const GameDef& gamedef, AssignedCmds* assigned_cmds);

  bool scout(const GameEnv& env,
             const Player& player,
             AssignedCmds* assigned_cmds);

  std::set<PointF> _place_taken;
};
