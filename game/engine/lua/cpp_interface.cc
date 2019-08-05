// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#include "cpp_interface.h"

void AttackRuleBook::Init(const std::string& lua_files) {
  init(lua_files, "attack_rule_book.lua");
}

bool AttackRuleBook::CanAttack(UnitType unit, UnitType target) {
  bool ret = false;
  Invoke(
      "attack_rule_book",
      "can_attack",
      &ret,
      static_cast<int>(unit),
      static_cast<int>(target));
  return ret;
}

void reg_engine_cpp_interfaces(const std::string& lua_files) {
  AttackRuleBook::Init(lua_files);
}
