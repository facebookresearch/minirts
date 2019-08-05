// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include "engine/mc_strategy_actor.h"

#include "ai/trainable_ai.h"
#include "ai/rule_extractor.h"

using TrainableRuleAI_ =  TrainableAI<RuleExtractor>;

class TrainableRuleAI : public TrainableRuleAI_ {
 public:
  TrainableRuleAI(
      const AIOption& opt,
      int threadId,
      std::shared_ptr<DataChannel> trainDc,
      std::shared_ptr<DataChannel> actDc)
      : TrainableRuleAI_(
            opt,
            threadId,
            trainDc,
            actDc,
            std::make_unique<RuleExtractor>(
                NUM_STRATEGY,
                opt.num_resource_bins,
                opt.resource_bin_size,
                GameDef::GetNumUnitType(),
                opt.use_moving_avg,
                opt.moving_avg_decay,
                opt.t_len)),
        numAction(NUM_STRATEGY)
  {}

  const int numAction;

 protected:
  bool assignCmds(
      const RTSStateExtend& state,
      Preload& preload,
      RTSAction* action) override {
    action->Init(getId(), numAction, RTSAction::RULE_BASED);
    action->SetAction(extractor_->action());

    if (!preload.Ok()) {
      return false;
    }

    // assert(numAction_ == (int)NUM_STRATEGY);
    MCStrategyActor strategy_actor(state.receiver(), preload, getId());
    return strategy_actor.ActByStrategy(state.env(), this, action);
  };

};
