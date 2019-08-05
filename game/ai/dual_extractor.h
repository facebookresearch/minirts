// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include "executor_extractor.h"

class DualExtractor {
 public:
  DualExtractor(
      int maxNumUnit,
      int numPrevCmds,
      int mapX,
      int mapY,
      int numResourceBins,
      int resourceBinSize,
      int numInstructions,
      int maxRawChars,
      int histLen,
      int numCmdTypes,
      int numUnitTypes,
      bool useMovingAvg,
      float movingAvgDecay,
      bool verbose)
      : normalExtractor_(
            maxNumUnit,
            numPrevCmds,
            mapX,
            mapY,
            numResourceBins,
            resourceBinSize,
            numInstructions,
            maxRawChars,
            histLen,
            numCmdTypes,
            numUnitTypes,
            useMovingAvg,
            movingAvgDecay,
            verbose),
        cheatExtractor_(
            maxNumUnit,
            numPrevCmds,
            mapX,
            mapY,
            numResourceBins,
            resourceBinSize,
            numInstructions,
            maxRawChars,
            histLen,
            numCmdTypes,
            numUnitTypes,
            useMovingAvg,
            movingAvgDecay,
            verbose,
            "nofow_")
  {}

  std::vector<std::shared_ptr<DataBlock>> getTrainSend() {
    auto blocks = normalExtractor_.getTrainSend();

    for (auto& f : cheatExtractor_.features_) {
      blocks.push_back(f.get().trajectory);
    }
    return blocks;
  }

  std::vector<std::shared_ptr<DataBlock>> getTrainReply() {
    return {};
  }

  std::vector<std::shared_ptr<DataBlock>> getActSend() {
    auto blocks = normalExtractor_.getActSend();

    for (auto& f : cheatExtractor_.features_) {
      blocks.push_back(f.get().buffer);
    }
    return blocks;
  }

  std::vector<std::shared_ptr<DataBlock>> getActReply() {
    return normalExtractor_.getActReply();
  }

  bool terminal() const {
    return normalExtractor_.terminal();
  }

  std::vector<int64_t> getRawInstruction() const {
    return normalExtractor_.getRawInstruction();
  }

  void writeCmds(
      const GameEnv& env,
      const CmdReceiver& receiver,
      PlayerId playerId,
      Preload* preload,
      std::map<UnitId, CmdBPtr>* assignedCmds,
      std::mt19937* rng) {
    normalExtractor_.writeCmds(env, receiver, playerId, preload, assignedCmds, rng);
  }

  bool skim(const RTSStateExtend& state, const AI& ai) {
    normalExtractor_.skim(state, ai);
    cheatExtractor_.skim(state, ai, true);
    return true;
  }

  void computeFeatures(
      Preload& preload,
      Preload& cheatPreload,
      const CmdReceiver& receiver,
      const GameEnv& env,
      PlayerId playerId,
      bool respectFow) {
    normalExtractor_.computeFeatures(preload, receiver, env, playerId, respectFow);
    cheatExtractor_.computeFeatures(cheatPreload, receiver, env, playerId, false);
  }

  // append current feature to trajectory
  void pushGameFeature() {
    normalExtractor_.pushGameFeature();
    cheatExtractor_.pushGameFeature();
  }

  void pushLastRAndTerminal() {
    normalExtractor_.pushLastRAndTerminal();
  }

  void pushActionAndPolicy() {
    normalExtractor_.pushActionAndPolicy();
  }

  // update some features, such as prevAction
  void postActUpdate() {
    normalExtractor_.postActUpdate();
    if (!normalExtractor_.instruction_.sameInstruction()) {
      cheatExtractor_.baseCountFeat_.getBuffer().copy_(
          cheatExtractor_.countFeat_.getBuffer());
    }
  }

  void updatePrevCmd(const std::map<UnitId, CmdBPtr>& assignedCmds) {
    normalExtractor_.updatePrevCmd(assignedCmds);
    cheatExtractor_.updatePrevCmd(assignedCmds);
  }

  // called after each act step
  void reset() {
    normalExtractor_.reset();
    cheatExtractor_.reset();
  }

  // called before each new game
  void newGame() {
    normalExtractor_.newGame();
    cheatExtractor_.newGame();
  }

 private:
  ExecutorExtractor normalExtractor_;
  ExecutorExtractor cheatExtractor_;
};
