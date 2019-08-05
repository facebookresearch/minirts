// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include <limits>

#include "cmd.h"
#include "common.h"
#include "unit_template.h"

class GameEnv;
class RTSMap;
class RTSGameOption;
class RuleActor;

// GameDef class. Have different implementations for different games.
class GameDef {
 public:
  std::vector<UnitTemplate> _units;

  // Initialize everything.
  void Init(const std::string& lua_files);

  static void GlobalInit(const std::string& lua_files);

  static int GetMapX();
  static int GetMapY();

  static int GetNumUnitType();
  static int GetNumAction();
  static int GetNumStrategy();
  static int GetMaxNumTownHall();

  static bool IsUnitTypeBuilding(UnitType t);
  static bool IsUnitTypeOffensive(UnitType t);

  bool CanAttack(UnitType attacker, UnitType target) const;

  bool HasBase() const;
  bool CanBuildAnyUnit(int amount) const;

  const UnitTemplate& unit(UnitType t) const {
    if (t < 0 || t >= (int)_units.size()) {
      std::cout << "UnitType " << t << " is not found!" << std::endl;
      throw std::range_error("Unit type is not found!");
    }
    return _units[t];
  }

  UnitType GetBuildFrom(UnitType t) const {
    return unit(t).GetBuildFrom();
  }

  // Get game initialization commands.
  std::vector<std::pair<CmdBPtr, int>> GetInitCmds(const RTSGameOption& option,
                                                   int seed) const;

  // Check winner for the game.
  PlayerId CheckWinner(const GameEnv& env) const;

  // Get implementation for different befaviors on dead units.
  void CmdOnDeadUnitImpl(GameEnv* env,
                         CmdReceiver* receiver,
                         UnitId _id,
                         UnitId _target) const;
};
