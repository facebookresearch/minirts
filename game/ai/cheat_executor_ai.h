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
#include "engine/mc_rule_actor.h"

#include "ai/ai.h"
#include "ai/cmd_reply.h"
#include "ai/dual_extractor.h"
#include "ai/trainable_ai.h"

using CheatExecutorAI_ = TrainableAI<DualExtractor>;

class CheatExecutorAI : public CheatExecutorAI_ {
 public:
  CheatExecutorAI(
      const AIOption& opt,
      int threadId,
      std::shared_ptr<DataChannel> trainDc,
      std::shared_ptr<DataChannel> actDc)
      : CheatExecutorAI_(
            opt,
            threadId,
            trainDc,
            actDc,
            std::make_unique<DualExtractor>(
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

  bool act(const RTSStateExtend& state, RTSAction* action) override {
    if (episodeStart_) {
      extractor_->newGame();
    }
    extractor_->skim(state, *this);

    const GameEnv& env = state.env();
    const CmdReceiver& receiver = state.receiver();
    const PlayerId playerId = getId();

    Preload preload;
    Preload cheatPreload;

    preload.GatherInfo(env, playerId, receiver, buildQueue, respectFow());
    cheatPreload.GatherInfo(env, playerId, receiver, buildQueue, false);
    extractor_->computeFeatures(preload, cheatPreload, receiver, env, playerId, respectFow());

    if (requireTrain_ && !episodeStart_) {
      extractor_->pushLastRAndTerminal();
    }

    if (requireTrain_ && stepIdx_ == option_.t_len) {
      extractor_->pushGameFeature();
      stepIdx_ = 0;
      trainDispatcher_->dispatch();
    }

    // act, get action from python actor
    // we need to send back the last_r even when the game terminated
    actDispatcher_->dispatch();

    if (extractor_->terminal()) {
      episodeStart_ = true;
      return false;
    }
    episodeStart_ = false;

    // push game feature immediately
    if (requireTrain_) {
      extractor_->pushGameFeature();
      stepIdx_ += 1;
    }
    extractor_->postActUpdate();

    bool success = assignCmds(state, preload, action);

    // push action and policy
    if (requireTrain_) {
      extractor_->pushActionAndPolicy();
    }

    extractor_->reset();
    return success;
  }

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
