// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include "utils/lua/interface.h"

#include "utils.h"

#include "engine/cmd_receiver.h"
#include "engine/cmd_specific.gen.h"
#include "engine/common.h"
#include "engine/gamedef.h"
#include "engine/map.h"
#include "engine/unit.h"

namespace detail {

struct _LuaStateProxy : public LuaClassInterface<StateProxy> {
  static void Init();
};

struct _LuaCoord : public LuaClassInterface<Coord> {
  static void Init();
};

struct _LuaRTSMap : public LuaClassInterface<RTSMap> {
  static void Init();
};

struct _LuaMapSlot : public LuaClassInterface<MapSlot> {
  static void Init();
};

struct _LuaTerrain : public LuaEnumInterface<Terrain> {
  static void Init();
};

struct _LuaUnitType : public LuaEnumInterface<UnitType> {
  static void Init();
};

struct _LuaUnitAttr : public LuaEnumInterface<UnitAttr> {
  static void Init();
};

struct _LuaCDType : public LuaEnumInterface<CDType> {
  static void Init();
};

struct _LuaCmdType : public LuaEnumInterface<CmdType> {
  static void Init();
};

struct _LuaPointF : public LuaClassInterface<PointF> {
  static void Init();
};

struct _LuaUnit : public LuaClassInterface<Unit> {
  static void Init();
};

struct _LuaUnitProperty : public LuaClassInterface<UnitProperty> {
  static void Init();
};

struct _LuaUnitTemplate : public LuaClassInterface<UnitTemplate> {
  static void Init();
};

struct _LuaCooldown : public LuaClassInterface<Cooldown> {
  static void Init();
};

struct _LuaBuildSkill : public LuaClassInterface<BuildSkill> {
  static void Init();
};

} // namespace detail

void reg_engine_lua_interfaces();
