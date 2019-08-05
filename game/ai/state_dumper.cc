// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#include "state_dumper.h"

bool isInstruction(const CmdBPtr& cmd) {
  auto cmdType = cmd->type();
  if (cmdType == ACCEPT_INSTRUCTION
      || cmdType == FINISH_INSTRUCTION
      || cmdType == INTERRUPT_INSTRUCTION
      || cmdType == WARN_INSTRUCTION
      || cmdType == ISSUE_INSTRUCTION) {
    // std::cout << "skip due to useless control instruction" << std::endl;
    return true;
  }
  return false;
}

bool StateDumper::processInstruction(const CmdBPtr& cmd) {
  assert(isInstruction(cmd));
  // // return true if this cmd is instruction related
  // // skip control instructions, except ISSUE_INSTRUCTION
  // if (cmd->type() == ACCEPT_INSTRUCTION
  //     || cmd->type() == FINISH_INSTRUCTION
  //     || cmd->type() == INTERRUPT_INSTRUCTION
  //     || cmd->type() == WARN_INSTRUCTION) {
  //   // std::cout << "skip due to useless control instruction" << std::endl;
  //   return true;
  // }

  if (cmd->type() == ISSUE_INSTRUCTION) {
    // process ISSUE_INSTRUCTION
    const auto* cmd_issue = dynamic_cast<const CmdIssueInstruction*>(cmd.get());
    assert(cmd_issue != nullptr);
    // it's id cannot be extraced from unit id
    if (cmd_issue->player_id() == getId()) {
      // std::cout << "tick: " << cmd->tick()
      //           << ", New instruction: " << cmd_issue->instruction() << std::endl;
      nextInstruction_ = cmd_issue->instruction();
    }
    return true;
  }
  return false;
}

bool StateDumper::act(const RTSStateExtend& state, RTSAction*) {
  Tick tick = state.GetTick();
  if (tick + 1 < option_.fs) {
    // before the first step, nothing to do
    return false;
  }

  Tick baseTick = tick - (tick + 1) % option_.fs;

  // compute preload
  const auto& env = state.env();
  const auto& receiver = state.receiver();
  Preload preload;
  preload.GatherInfo(env, getId(), receiver, buildQueue, respectFow());
  // when replay has ended or game has ended
  if (receiver.GetForceTerminate() || !preload.Ok()) {
    return false;
  }
  // // when game has ended
  // if (!preload.Ok()) {
  //   return false;
  // }
  preload.setUnitId2IdxMaps();
  GameEnvAspect aspect(env, getId(), respectFow());

  // check instruction first
  const auto& loadedReplay = replayLoader_.GetLoadedReplay();
  {
    size_t nextReplayIdx = nextReplayIdx_;
    while (nextReplayIdx < loadedReplay.size()) {
      const auto& cmd = loadedReplay[nextReplayIdx];
      auto cmdTick = cmd->tick();

      if (cmdTick > tick) {
        // std::cout << "break" << std::endl;
        break;
      }
      assert(tick < option_.fs || cmdTick == tick);
      ++nextReplayIdx;

      // this cmd is instruction-related
      if (!isInstruction(cmd)) {
        continue;
      }
      processInstruction(cmd);
    }
  }

  // dump game state
  if ((tick + 1) % option_.fs == 0) {
    // std::cout << "tick: " << tick << ", dumping state" << std::endl;
    // update instruction
    if (!nextInstruction_.empty()) {
      currentInstruction_ = nextInstruction_;
      nextInstruction_ = "";
    }

    nlohmann::json data = preload.log2Json(aspect, receiver);
    data["instruction"] = currentInstruction_;
    data["map"] = env.LogMap2Json(getId());
    logState(tick, std::move(data));
  }

  // go over cmds other than instructions
  std::list<UnitId> unitWithTarget;
  while (nextReplayIdx_ < loadedReplay.size()) {
    const auto& cmd = loadedReplay[nextReplayIdx_];
    auto cmdTick = cmd->tick();

    if (cmdTick > tick) {
      // std::cout << "break" << std::endl;
      break;
    }
    assert(tick < option_.fs || cmdTick == tick);
    ++nextReplayIdx_;

    if (cmdTick + 1 < option_.fs) {
      // before the first step, ignore cmds
      continue;
    }

    // this cmd is instruction-related
    if (isInstruction(cmd)) {
      continue;
    }

    if (!nextInstruction_.empty()) {
      // TODO: do we need drop them at all???
      continue;
    }

    // process everything else
    const auto playerId = ExtractPlayerId(cmd->id());
    if (playerId != getId()) {
      continue;
    }

    // std::cout << "cmd tick: " << cmdTick << std::endl;
    // std::cout << ">>>cmd: " << cmd->PrintInfo() << std::endl;

    const CmdDurative* durative = dynamic_cast<const CmdDurative*>(cmd.get());
    assert(durative != nullptr);

    CmdTarget cmdTarget = CreateCmdTarget(aspect, durative);
    // target can be idle due to missing attack target
    // only dump if target is not idle
    if (cmdTarget.cmdType != CMD_TARGET_IDLE) {
      auto cmdJson = cmdTarget.log2Json(preload.getUnitId2Idx());
      cmdJson["tick"] = durative->tick();
      appendTargetJson(baseTick, std::move(cmdJson));

      // dump partial state
      unitWithTarget.push_back(cmdTarget.unitId);
    }
  }

  if (!unitWithTarget.empty()) {
    // std::cout << "patching cmd from " << tick << " to " << baseTick << std::endl;
    auto patch = preload.partialLog2Json(aspect, receiver, unitWithTarget);
    patchMyUnits(baseTick, std::move(patch));
  }

  return true;
}
