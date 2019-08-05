// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include <nlohmann/json.hpp>

#include "cmd.h"
#include "cmd_receiver.h"
#include "cmd_specific.gen.h"
#include "custom_enum.h"
#include "game_env.h"
#include "gamedef.h"
#include "unit.h"

custom_enum(CmdTargetType,
            CMD_TARGET_IDLE = 0,
            CMD_TARGET_GATHER,
            CMD_TARGET_ATTACK,
            CMD_TARGET_BUILD_BUILDING,
            CMD_TARGET_BUILD_UNIT,
            CMD_TARGET_MOVE,
            CMD_TARGET_CONT,
            NUM_CMD_TARGET_TYPE);

CmdTargetType getCmdTargetType(const CmdBPtr& cmd);

class CmdTarget {
 public:
  CmdTarget(CmdTargetType cmdType,
            UnitId unitId,
            UnitId targetId,
            UnitType targetType,
            float targetX,
            float targetY)
      : cmdType(cmdType)
      , unitId(unitId)
      , targetId(targetId)
      , targetType(targetType)
      , targetX(targetX)
      , targetY(targetY) {
  }

  nlohmann::json log2Json(const std::map<UnitId, int>& id2idx) const {
    nlohmann::json data;
    data["cmd_type"] = int(cmdType);
    data["unit_id"] = unitId;
    data["target_id"] = targetId;
    data["target_type"] = int(targetType);
    data["target_x"] = targetX;
    data["target_y"] = targetY;
    data["target_gather_idx"] = 0;
    data["target_attack_idx"] = 0;
    if (cmdType == CMD_TARGET_GATHER) {
      data["target_gather_idx"] = id2idx.at(targetId);
    } else if (cmdType == CMD_TARGET_ATTACK) {
      data["target_attack_idx"] = id2idx.at(targetId);
    }
    return data;
  }

  const CmdTargetType cmdType;
  const UnitId unitId;
  const UnitId targetId;
  const UnitType targetType;
  const float targetX;
  const float targetY;
};

CmdTarget CreateCmdTargetForUnit(const GameEnvAspect& aspect,
                                 const CmdReceiver& receiver,
                                 const Unit& u);

CmdTarget CreateCmdTarget(const GameEnvAspect& aspect, const CmdDurative* c);
