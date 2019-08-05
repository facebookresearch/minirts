// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include "utils/lua/interface.h"

#include "engine/cmd_receiver.h"
#include "engine/gamedef.h"
#include "engine/lua/utils.h"
#include "engine/map.h"

struct RTSMapGenerator : public CppClassInterface<RTSMapGenerator> {
  static void Init(const std::string& lua_files);

  static void Generate(RTSMap& map, int num_players, int seed);

  static void
  GenerateRandom(RTSMap& map, int num_players, int seed, bool no_terrain);
};

struct RTSUnitGenerator : public CppClassInterface<RTSUnitGenerator> {
  static void Init(const std::string& lua_files);

  static void
  Generate(RTSMap* map, int num_players, int seed, CmdReceiver* cmd_receiver);

  static void GenerateRandom(
      RTSMap* map,
      CmdReceiver* cmd_receiver,
      int num_players,
      int seed,
      int resource,
      int resource_dist,
      int num_resources,
      bool fair,
      int num_peasants,
      int num_extra_units);
};

struct RTSUnitFactory : public CppClassInterface<RTSUnitFactory> {
  static void Init(const std::string& lua_files);

  static UnitTemplate InitResource();

  static UnitTemplate InitPeasant();

  static UnitTemplate InitSwordman();

  static UnitTemplate InitSpearman();

  static UnitTemplate InitCavalry();

  static UnitTemplate InitArcher();

  static UnitTemplate InitDragon();

  static UnitTemplate InitCatapult();

  static UnitTemplate InitBarrack();

  static UnitTemplate InitBlacksmith();

  static UnitTemplate InitStable();

  static UnitTemplate InitWorkshop();

  static UnitTemplate InitAviary();

  static UnitTemplate InitArchery();

  static UnitTemplate InitGuardTower();

  static UnitTemplate InitTownHall();
};

void reg_cpp_interfaces(const std::string& lua_files);
