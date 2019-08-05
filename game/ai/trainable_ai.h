// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include "ai/ai.h"
#include "data_channel.h"
#include "dispatcher.h"

using tube::DataChannel;
using tube::Dispatcher;

template <typename Extractor>
class TrainableAI : public AI {
 public:
  TrainableAI(
      const AIOption& opt,
      int threadId,
      std::shared_ptr<DataChannel> trainDc,
      std::shared_ptr<DataChannel> actDc,
      std::unique_ptr<Extractor> extractor)
      : AI(opt, threadId),
        episodeStart_(true),
        requireTrain_(trainDc != nullptr),
        stepIdx_(0),
        extractor_(std::move(extractor))
  {
    if (trainDc != nullptr) {
      trainDispatcher_ = std::make_unique<Dispatcher>(std::move(trainDc));
      trainDispatcher_->addDataBlocks(extractor_->getTrainSend(), extractor_->getTrainReply());
    }

    assert(actDc != nullptr);
    actDispatcher_ = std::make_unique<Dispatcher>(std::move(actDc));
    actDispatcher_->addDataBlocks(extractor_->getActSend(), extractor_->getActReply());
  }

  // virtual bool skim(const RTSStateExtend& state) override;

  // virtual bool act(const RTSStateExtend& state, RTSAction* action) override;

 protected:
  // assign cmds to units
  virtual bool assignCmds(
      const RTSStateExtend& state, Preload& preload, RTSAction* action) = 0;

  bool episodeStart_;
  bool requireTrain_;
  int stepIdx_;

  std::unique_ptr<Extractor> extractor_;
  std::unique_ptr<Dispatcher> trainDispatcher_;
  std::unique_ptr<Dispatcher> actDispatcher_;
};

// template <typename Extractor>
// bool TrainableAI<Extractor>::skim(const RTSStateExtend& state) {
//   return extractor_->skim(state, *this);
// }

// template <typename Extractor>
// bool TrainableAI<Extractor>::act(const RTSStateExtend& state, RTSAction* action) {
//   const GameEnv& env = state.env();
//   const CmdReceiver& receiver = state.receiver();
//   const PlayerId playerId = getId();

//   Preload preload;
//   preload.GatherInfo(env, playerId, receiver, buildQueue, respectFow());
//   extractor_->computeFeatures(preload, receiver, env, playerId, respectFow());

//   if (requireTrain_ && !episodeStart_) {
//     extractor_->pushLastRAndTerminal();
//   }

//   if (requireTrain_ && stepIdx_ == option_.t_len) {
//     extractor_->pushGameFeature();
//     stepIdx_ = 0;
//     trainDispatcher_->dispatch();
//   }

//   // act, get action from python actor
//   // we need to send back the last_r even when the game terminated
//   actDispatcher_->dispatch();

//   if (extractor_->terminal()) {
//     episodeStart_ = true;
//     extractor_->newGame();
//     return false;
//   }
//   episodeStart_ = false;

//   // push game feature immediately
//   if (requireTrain_) {
//     extractor_->pushGameFeature();
//     stepIdx_ += 1;
//   }
//   extractor_->postActUpdate();

//   bool success = assignCmds(state, preload, action);

//   // push action and policy
//   if (requireTrain_) {
//     extractor_->pushActionAndPolicy();
//   }

//   extractor_->reset();
//   return success;
// }
