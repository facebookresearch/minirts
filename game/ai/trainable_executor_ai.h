// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include "engine/utils.h"
#include "engine/cmd_target.h"
#include "engine/preload.h"
// #include "engine/python_options.h"
#include "engine/mc_rule_actor.h"

#include "ai/ai.h"
#include "ai/cmd_reply.h"
#include "ai/executor_extractor.h"

using TrainableExecutorAI_ = TrainableAI<ExecutorExtractor>;

class TrainableExecutorAI : public TrainableExecutorAI_ {
 public:
  TrainableExecutorAI(
      const AIOption& opt,
      int threadId,
      std::shared_ptr<DataChannel> trainDc,
      std::shared_ptr<DataChannel> actDc)
      : TrainableExecutorAI_(
            opt,
            threadId,
            trainDc,
            actDc,
            std::make_unique<ExecutorExtractor>(
                opt.max_num_units,
                opt.num_prev_cmds,
                GameDef::GetMapX(),
                GameDef::GetMapY(),
                opt.num_resource_bins,
                opt.resource_bin_size,
                opt.num_instructions,
                opt.max_raw_chars,
                opt.t_len,
                NUM_CMD_TARGET_TYPE,
                GameDef::GetNumUnitType(),
                opt.use_moving_avg,
                opt.moving_avg_decay,
                opt.verbose))
  {}

 protected:
  bool assignCmds(
      const RTSStateExtend& state,
      Preload& preload,
      RTSAction* action) override {
    action->Init(getId(), 0, RTSAction::INSTRUCTION_BASED);

    if (!preload.Ok()) {
      return false;
    }

    const GameEnv& env = state.env();
    const CmdReceiver& receiver = state.receiver();

    extractor_->writeCmds(
        env, receiver, getId(), &preload, &(action->cmds()), &rng);

    // TODO: keep or discard?
    MCRuleActor ruleActor(receiver, preload, getId());
    ruleActor.SetTowerAutoAttack(env, &(action->cmds()));

    extractor_->updatePrevCmd(action->cmds());
    action->SetInstruction(extractor_->getRawInstruction());
    return true;
  }
};
