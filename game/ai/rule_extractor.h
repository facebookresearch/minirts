// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include "data_block.h"

#include "engine/preload.h"
#include "engine/game_state_ext.h"
#include "ai/ai.h"

using tube::DataBlock;
using tube::FixedLengthTrajectory;

// feature extractor for rule based trained AI
class RuleExtractor {
 public:
  RuleExtractor(
      int numAction,
      int numResourceBin,
      int resourceBinSize,
      int numUnitType,
      bool useMovingAvg,
      float movingAvgDecay,
      int tLen)
      : numAction(numAction),
        numResourceBin(numResourceBin),
        resourceBinSize(resourceBinSize),
        numUnitType(numUnitType),
        useMovingAvg(useMovingAvg),
        movingAvgDecay(movingAvgDecay),
        unitCount_("unit_count", tLen + 1, {numUnitType * 2}, torch::kFloat32),
        unitConsCount_("unit_cons_count", tLen + 1, {numUnitType}, torch::kFloat32),
        movingEnemyCount_("moving_enemy_count", tLen + 1, {numUnitType}, torch::kFloat32),
        resourceBin_("resource_bin", tLen + 1, {numResourceBin}, torch::kFloat32),
        prevAction_("prev_action", tLen + 1, {numAction}, torch::kInt64),
        action_("action", tLen, {numAction}, torch::kInt64),
        policy_("policy", tLen, {numAction}, torch::kFloat32),
        reward_("reward", tLen, {1}, torch::kFloat32),
        terminal_("terminal", tLen, {1}, torch::kFloat32)
  {}

  std::vector<std::shared_ptr<DataBlock>> getTrainSend() {
    std::vector<std::shared_ptr<DataBlock>> blocks = {
      unitCount_.trajectory,
      unitConsCount_.trajectory,
      movingEnemyCount_.trajectory,
      resourceBin_.trajectory,
      prevAction_.trajectory,
      action_.trajectory,
      policy_.trajectory,
      reward_.trajectory,
      terminal_.trajectory
    };
    return blocks;
  }

  std::vector<std::shared_ptr<DataBlock>> getTrainReply() {
    return std::vector<std::shared_ptr<DataBlock>>();
  }

  std::vector<std::shared_ptr<DataBlock>> getActSend() {
    std::vector<std::shared_ptr<DataBlock>> blocks = {
      unitCount_.buffer,
      unitConsCount_.buffer,
      movingEnemyCount_.buffer,
      resourceBin_.buffer,
      prevAction_.buffer,
      reward_.buffer,
      terminal_.buffer
    };
    return blocks;
  }

  std::vector<std::shared_ptr<DataBlock>> getActReply() {
    std::vector<std::shared_ptr<DataBlock>> blocks = {
      action_.buffer,
      policy_.buffer,
    };
    return blocks;
  }

  bool terminal() const {
    return terminal_.getBuffer().item<float>() > 0;
  }

  std::vector<int64_t> action() const {
    std::vector<int64_t> action(numAction, 0);
    auto accessor = action_.getBuffer().accessor<int64_t, 1>();
    for (int i = 0; i < numAction; ++i) {
      action[i] = accessor[i];
    }
    return action;
    // return action_.getData();
  }

  // called after each act step
  void reset() {}

  // called before each new game
  void newGame();

  bool skim(const RTSStateExtend& state, const AI& ai);

  void computeFeatures(
      const Preload& preload,
      const CmdReceiver& receiver,
      const GameEnv& env,
      PlayerId playerId,
      bool respectFow);

  void postActUpdate();

  // append current feature to trajectory
  int pushGameFeature();

  int pushLastRAndTerminal();

  int pushActionAndPolicy();

  const int numAction;
  const int numResourceBin;
  const int resourceBinSize;
  const int numUnitType;
  const bool useMovingAvg;
  const float movingAvgDecay;

 private:
  void computeUnitCount(const Preload& preload);

  void computeUnitConsCount(const Preload& preload);

  void computeResourceBin(int resource);

  FixedLengthTrajectory unitCount_;
  FixedLengthTrajectory unitConsCount_;
  FixedLengthTrajectory movingEnemyCount_;
  FixedLengthTrajectory resourceBin_;
  FixedLengthTrajectory prevAction_;
  FixedLengthTrajectory action_;
  FixedLengthTrajectory policy_;
  FixedLengthTrajectory reward_;
  FixedLengthTrajectory terminal_;
};
