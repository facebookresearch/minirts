// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include "utils/lua/interface.h"
#include "engine/common.h"

#include <string>

struct AttackRuleBook : public CppClassInterface<AttackRuleBook> {
  static void Init(const std::string& lua_files);

  static bool CanAttack(UnitType unit, UnitType target);
};

void reg_engine_cpp_interfaces(const std::string& lua_files);
