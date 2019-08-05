// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#include "engine/gamedef.h"
#include "engine/cmd.gen.h"
#include "engine/cmd_specific.gen.h"
#include "engine/game_env.h"
#include "engine/lua/cpp_interface.h"
#include "engine/lua/lua_interface.h"
#include "engine/mc_strategy_actor.h"
#include "engine/rule_actor.h"
#include "game_MC/cmd_specific.gen.h"
#include "lua/cpp_interface.h"
#include "lua/lua_interface.h"

#include "ai/ai_factory.h"
#include "ai/rule_ai.h"
#include "ai/state_dumper.h"

// TODO: horrible design
int GameDef::GetMapX() {
  return 32;
}

int GameDef::GetMapY() {
  return 32;
}

int GameDef::GetNumUnitType() {
  return NUM_MINIRTS_UNITTYPE;
}

int GameDef::GetNumAction() {
  return NUM_AISTATE;
}

int GameDef::GetNumStrategy() {
  return NUM_STRATEGY;
}

int GameDef::GetMaxNumTownHall() {
  return 3;
}

bool GameDef::IsUnitTypeBuilding(UnitType t) {
  bool ret = (t == TOWN_HALL) || (t == RESOURCE) || (t == BARRACK) ||
             (t == BLACKSMITH) || (t == STABLE) || (t == WORKSHOP) ||
             (t == AVIARY) || (t == ARCHERY) || (t == GUARD_TOWER);
  return ret;
}

bool GameDef::IsUnitTypeOffensive(UnitType t) {
  bool ret = (t != PEASANT && !GameDef::IsUnitTypeBuilding(t));
  return ret;
}

bool GameDef::CanAttack(UnitType attacker, UnitType target) const {
  float attack_mult = _units[attacker].GetAttackMultiplier(target);
  return attack_mult > 0;
}

bool GameDef::CanBuildAnyUnit(int amount) const {
  for (const auto& unit : _units) {
    if (unit.GetUnitCost() <= amount) {
      return true;
    }
  }
  return false;
}

void GameDef::GlobalInit(const std::string& lua_files) {
  // using AIFactory_ = AIFactory<AI>;

  reg_engine();
  reg_engine_specific();
  reg_minirts_specific();
  // reg_engine_cmd_lua(lua_files);
  reg_engine_lua_interfaces();
  reg_engine_cpp_interfaces(lua_files);
  reg_lua_interfaces();
  reg_cpp_interfaces(lua_files);

  // hack: use seed as threadId for randomness
  AIFactory::RegisterAI("dumper", [](int, const Params& params) {
    AIOption ai_option;
    ai_option.log_state = true;
    ai_option.fs = GetIntParam(params, "fs", 1);
    ai_option.fow = GetIntParam(params, "fow", 1);

    auto replay = GetStringParam(params, "replay", "");
    return new StateDumper(ai_option, replay);
  });

  AIFactory::RegisterAI("simple", [](int seed, const Params& params) {
    AIOption ai_option;
    ai_option.resource_scale = GetFloatParam(params, "resource_scale", 1.0);
    ai_option.log_state = GetBoolParam(params, "log_state", false);
    ai_option.fs = GetIntParam(params, "fs", 1);

    auto unit_type = GetUnitTypeParam(params, "unit_type", INVALID_UNITTYPE);
    return new SimpleAI(ai_option, seed, unit_type);
  });

  AIFactory::RegisterAI("medium", [](int seed, const Params& params) {
    AIOption ai_option;
    ai_option.resource_scale = GetFloatParam(params, "resource_scale", 1.0);
    ai_option.log_state = GetBoolParam(params, "log_state", false);
    ai_option.fs = GetIntParam(params, "fs", 1);

    auto unit_type = GetUnitTypeParam(params, "unit_type", INVALID_UNITTYPE);
    auto build_tower = GetBoolParam(params, "tower", false);
    return new MediumAI(ai_option, seed, unit_type, build_tower);
  });

  // AIFactory::RegisterAI("expansion", [](int seed, const Params& params) {
  //     AIOption ai_option;
  //     ai_option.resource_scale = GetFloatParam(params, "resource_scale",
  //     1.0);
  //     ai_option.log_state = GetBoolParam(params, "log_state", false);
  //     ai_option.fs = GetIntParam(params, "fs", 1);

  //     return new ExpansionAI(ai_option, seed);
  // });

  AIFactory::RegisterAI("strong", [](int seed, const Params& params) {
    AIOption ai_option;
    ai_option.resource_scale = GetFloatParam(params, "resource_scale", 1.0);
    ai_option.log_state = GetBoolParam(params, "log_state", false);
    ai_option.fs = GetIntParam(params, "fs", 1);

    return new StrongAI(ai_option, seed);
  });

  // AIFactory::RegisterAI("two_units_rush", [](int seed, const Params& params)
  // {
  //     AIOption ai_option;
  //     ai_option.resource_scale = GetFloatParam(params, "resource_scale",
  //     1.0);
  //     ai_option.log_state = GetBoolParam(params, "log_state", false);
  //     ai_option.fs = GetIntParam(params, "fs", 1);

  //     auto unit_type1 = GetUnitTypeParam(params, "unit_type1",
  //     INVALID_UNITTYPE);
  //     auto unit_type2 = GetUnitTypeParam(params, "unit_type2",
  //     INVALID_UNITTYPE);
  //     auto build_tower = GetBoolParam(params, "tower", false);
  //     return new TwoUnitsRushAI(ai_option, seed, unit_type1, unit_type2,
  //     build_tower);
  // });

  // AIFactory::RegisterAI("tower_rush", [](int seed, const Params& params) {
  //     AIOption ai_option;
  //     ai_option.resource_scale = GetFloatParam(params, "resource_scale",
  //     1.0);
  //     ai_option.log_state = GetBoolParam(params, "log_state", false);
  //     ai_option.fs = GetIntParam(params, "fs", 1);

  //     return new TowerRushAI(ai_option, seed);
  // });

  // AIFactory::RegisterAI("peasant_rush", [](int seed, const Params& params) {
  //     AIOption ai_option;
  //     ai_option.resource_scale = GetFloatParam(params, "resource_scale",
  //     1.0);
  //     ai_option.log_state = GetBoolParam(params, "log_state", false);
  //     ai_option.fs = GetIntParam(params, "fs", 1);

  //     return new PeasantRushAI(ai_option, seed);
  // });

  // AIFactory::RegisterAI("onboard", [](int seed, const Params&) {
  //   AIOption ai_option;
  //   return new OnboardAI(ai_option, seed);
  // });

  // AIFactory::RegisterAI("coach", [](int seed, const Params&) {
  //   AIOption ai_option;
  //   return new ScriptedCoachAI(ai_option, seed);
  // });

  // AIFactory::RegisterAI("test_build", [](int, const Params&) {
  //   AIOption ai_option;
  //   return new TestBuildAI(ai_option, 0);
  // });

  // AIFactory::RegisterAI("test_attack", [](int, const Params&) {
  //   AIOption ai_option;
  //   return new TestAttackAI(ai_option, 0);
  // });

  // AIFactory::RegisterAI("test_build_queue", [](int, const Params&) {
  //   AIOption ai_option;
  //   return new TestBuildQueueAI(ai_option, 0);
  // });

  // AIFactory::RegisterAI("test_void", [](int, const Params&) {
  //   AIOption ai_option;
  //   return new TestVoidAI(ai_option, 0);
  // });
}

void GameDef::Init(const std::string& lua_files) {
  reg_engine();
  reg_engine_specific();
  reg_minirts_specific();

  reg_engine_lua_interfaces();
  reg_engine_cpp_interfaces(lua_files);
  reg_lua_interfaces();
  reg_cpp_interfaces(lua_files);

  _units.assign(GetNumUnitType(), UnitTemplate());

  _units[RESOURCE] = RTSUnitFactory::InitResource();
  _units[PEASANT] = RTSUnitFactory::InitPeasant();
  _units[SWORDMAN] = RTSUnitFactory::InitSwordman();
  _units[SPEARMAN] = RTSUnitFactory::InitSpearman();
  _units[CAVALRY] = RTSUnitFactory::InitCavalry();
  _units[ARCHER] = RTSUnitFactory::InitArcher();
  _units[DRAGON] = RTSUnitFactory::InitDragon();
  _units[CATAPULT] = RTSUnitFactory::InitCatapult();
  _units[BARRACK] = RTSUnitFactory::InitBarrack();
  _units[BLACKSMITH] = RTSUnitFactory::InitBlacksmith();
  _units[STABLE] = RTSUnitFactory::InitStable();
  _units[WORKSHOP] = RTSUnitFactory::InitWorkshop();
  _units[AVIARY] = RTSUnitFactory::InitAviary();
  _units[ARCHERY] = RTSUnitFactory::InitArchery();
  _units[GUARD_TOWER] = RTSUnitFactory::InitGuardTower();
  _units[TOWN_HALL] = RTSUnitFactory::InitTownHall();
}

std::vector<std::pair<CmdBPtr, int>> GameDef::GetInitCmds(
    const RTSGameOption& option, int seed) const {
  std::vector<std::pair<CmdBPtr, int>> init_cmds;
  auto map_gen = CmdBPtr(new CmdGenerateMap(INVALID, seed, option.no_terrain));
  init_cmds.push_back(make_pair(std::move(map_gen), 1));
  auto unit_gen = CmdBPtr(new CmdGenerateUnit(INVALID,
                                              seed,
                                              option.resource,
                                              option.resource_dist,
                                              option.num_resources,
                                              option.fair,
                                              option.num_peasants,
                                              option.num_extra_units));
  init_cmds.push_back(make_pair(std::move(unit_gen), 2));

  if (option.team_play) {
    // freeze the game right after we generate units, i.e. using the same tick
    init_cmds.push_back(
        make_pair(CmdBPtr(new CmdFreezeGame(INVALID, true)), 2));
  }
  return init_cmds;
}

PlayerId GameDef::CheckWinner(const GameEnv& env) const {
  const auto town_hall_player_id = env.CheckTownHall();
  if (town_hall_player_id != INVALID) {
    return town_hall_player_id;
  }
  return env.CheckUnitsAndMoney();
}

void GameDef::CmdOnDeadUnitImpl(GameEnv* env,
                                CmdReceiver* receiver,
                                UnitId /*_id*/,
                                UnitId _target) const {
  Unit* target = env->GetUnit(_target);
  if (target == nullptr) {
    return;
  }
  receiver->SendCmd(CmdIPtr(new CmdRemove(_target)));
}
