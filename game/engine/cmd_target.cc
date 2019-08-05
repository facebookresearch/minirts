// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#include "cmd_target.h"

CmdTargetType getCmdTargetType(const CmdBPtr& cmd) {
  CmdType cmdType = cmd->type();
  if (cmdType == GATHER) {
    return CMD_TARGET_GATHER;
  }
  if (cmdType == ATTACK) {
    return CMD_TARGET_ATTACK;
  }
  if (cmdType == MOVE) {
    return CMD_TARGET_MOVE;
  }
  if (cmdType == BUILD) {
    const CmdBuild* build = dynamic_cast<const CmdBuild*>(cmd.get());
    assert(build != nullptr);
    if (GameDef::IsUnitTypeBuilding(build->build_type())) {
      return CMD_TARGET_BUILD_BUILDING;
    } else {
      return CMD_TARGET_BUILD_UNIT;
    }
  }
  std::cout << "Warning: unknown type conversion" << cmdType << std::endl;
  assert(false);
}

CmdTarget CreateCmdTargetForUnit(const GameEnvAspect& aspect,
                                 const CmdReceiver& receiver,
                                 const Unit& u) {
  UnitId unitId = u.GetId();
  const CmdDurative* c = receiver.GetUnitDurativeCmd(unitId);
  if (c == nullptr) {
    return CmdTarget(
        CMD_TARGET_IDLE, unitId, 0, static_cast<UnitType>(0), 0, 0);
  } else {
    return CreateCmdTarget(aspect, c);
  }
}

CmdTarget CreateCmdTarget(const GameEnvAspect& aspect, const CmdDurative* c) {
  assert(c != nullptr);
  CmdTargetType cmdTargetType = CMD_TARGET_IDLE;
  UnitId targetId = 0;
  UnitType targetType = static_cast<UnitType>(0);
  float targetX = 0;
  float targetY = 0;

  CmdType cmdType = c->type();
  if (cmdType == GATHER) {
    const CmdGather* cmd = dynamic_cast<const CmdGather*>(c);
    assert(cmd != nullptr);
    UnitId targetId_ = cmd->resource();
    const Unit* target = aspect.GetUnit(targetId_);
    if (target != nullptr) {
      targetId = targetId_;
      cmdTargetType = CMD_TARGET_GATHER;
      targetType = RESOURCE;
    }
  } else if (cmdType == ATTACK) {
    const CmdAttack* cmd = dynamic_cast<const CmdAttack*>(c);
    assert(cmd != nullptr);
    UnitId targetId_ = cmd->target();
    const Unit* target = aspect.GetUnit(targetId_);
    // only if the target is still alive
    // TODO: this is not perfect, we should remove the attack cmd
    // from the durative cmd queue if the target is dead
    if (target != nullptr) {
      targetId = targetId_;
      cmdTargetType = CMD_TARGET_ATTACK;
      targetType = target->GetUnitType();
    }
  } else if (cmdType == BUILD) {
    const CmdBuild* cmd = dynamic_cast<const CmdBuild*>(c);
    assert(cmd != nullptr);
    targetType = cmd->build_type();
    if (GameDef::IsUnitTypeBuilding(targetType)) {
      cmdTargetType = CMD_TARGET_BUILD_BUILDING;
      targetX = cmd->p().x;
      targetY = cmd->p().y;
    } else {
      cmdTargetType = CMD_TARGET_BUILD_UNIT;
    }
  } else if (cmdType == MOVE) {
    const CmdMove* cmd = dynamic_cast<const CmdMove*>(c);
    assert(cmd != nullptr);
    cmdTargetType = CMD_TARGET_MOVE;
    targetX = cmd->p().x;
    targetY = cmd->p().y;
  } else {
    std::cout << "Error: Unexpected CmdType: " << cmdType << std::endl;
    assert(false);
  }
  return CmdTarget(
      cmdTargetType, c->id(), targetId, targetType, targetX, targetY);
}
