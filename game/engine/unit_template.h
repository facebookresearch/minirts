// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include "unit_property.h"

class UnitTemplate {
 private:
  // TODO: init them to proper value
  UnitProperty _property;
  std::set<CmdType> _allowed_cmds;
  std::vector<float> _attack_multiplier;
  std::vector<UnitType> _attack_order;
  std::vector<int> _unit_type_to_attack_order;
  std::vector<bool> _cant_move_over;
  UnitType _build_from;
  int _build_cost;
  std::vector<BuildSkill> _build_skills;

 public:
  UnitTemplate()
      : _attack_multiplier(NUM_MINIRTS_UNITTYPE, 0.0)
      , _unit_type_to_attack_order(
            NUM_MINIRTS_UNITTYPE, std::numeric_limits<int>::max())
      , _cant_move_over(NUM_TERRAIN, false) {
  }

  void SetProperty(UnitProperty prop) {
    _property = prop;
  }

  const UnitProperty& GetProperty() const {
    return _property;
  }

  void SetUnitCost(int cost) {
    _build_cost = cost;
  }

  void SetUnitHP(int hp) {
    _property._hp = hp;
    _property._max_hp = hp;
  }

  int GetUnitCost() const {
    return _build_cost;
  }

  bool CmdAllowed(CmdType cmd) const {
    if (cmd == CMD_DURATIVE_LUA)
      return true;
    return _allowed_cmds.find(cmd) != _allowed_cmds.end();
  }

  float GetAttackMultiplier(UnitType unit_type) const {
    return _attack_multiplier[unit_type];
  }

  const std::vector<float>& GetAllAttackMultiplier() const {
    return _attack_multiplier;
  }

  const std::vector<UnitType>& GetAttackOrder() const {
    return _attack_order;
  }

  const std::vector<int>& GetUnitType2AttackOrder() const {
    return _unit_type_to_attack_order;
  }

  bool CanMoveOver(Terrain terrain) const {
    return !_cant_move_over[terrain];
  }

  const std::vector<BuildSkill>& GetBuildSkills() const {
    return _build_skills;
  }

  UnitType GetUnitTypeFromHotKey(char hotkey) const {
    for (const auto& skill : _build_skills) {
      if (hotkey == skill.GetHotKey()[0]) {
        return skill.GetUnitType();
      }
    }
    return INVALID_UNITTYPE;
  }

  void AddAllowedCmd(int cmd) {
    _allowed_cmds.insert(static_cast<CmdType>(cmd));
  }

  const std::set<CmdType>& GetAllowedCmds() const {
    return _allowed_cmds;
  }

  void SetAttackMultiplier(int unit_type, double mult) {
    _attack_multiplier[unit_type] = static_cast<float>(mult);
  }

  void AppendAttackOrder(int unit_type) {
    _attack_order.push_back(static_cast<UnitType>(unit_type));
    _unit_type_to_attack_order[unit_type] = (int)_attack_order.size();
  }

  void AddCantMoveOver(int terrain) {
    _cant_move_over[terrain] = true;
  }

  void SetBuildFrom(int unit_type) {
    _build_from = (UnitType)unit_type;
  }

  UnitType GetBuildFrom() const {
    return _build_from;
  }

  void AddBuildSkill(BuildSkill skill) {
    _build_skills.push_back(std::move(skill));
  }
};
