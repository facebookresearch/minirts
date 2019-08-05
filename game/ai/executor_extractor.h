// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include "engine/game_state.h"
#include "engine/preload.h"
#include "ai/ai.h"

#include "ai/coach_instruction.h"
#include "ai/cmd_reply.h"

using tube::DataBlock;
using tube::FixedLengthTrajectory;

class ExecutorExtractor {
 public:
  ExecutorExtractor(
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
      bool verbose,
      std::string prefix="");

  std::vector<std::shared_ptr<DataBlock>> getTrainSend() {
    std::vector<std::shared_ptr<DataBlock>> blocks;
    for (auto& f : features_) {
      blocks.push_back(f.get().trajectory);
    }
    for (auto& t : instruction_.getTrainSend()) {
      blocks.push_back(t);
    }
    blocks.push_back(reward_.trajectory);
    blocks.push_back(terminal_.trajectory);
    return blocks;
  }

  std::vector<std::shared_ptr<DataBlock>> getTrainReply() {
    return {};
  }

  std::vector<std::shared_ptr<DataBlock>> getActSend() {
    std::vector<std::shared_ptr<DataBlock>> blocks;
    for (auto& f : features_) {
      blocks.push_back(f.get().buffer);
    }
    for (auto& b : instruction_.getActSend()) {
      blocks.push_back(b);
    }
    blocks.push_back(reward_.buffer);
    return blocks;
  }

  std::vector<std::shared_ptr<DataBlock>> getActReply() {
    std::vector<std::shared_ptr<DataBlock>> blocks;
    for (auto& b : instruction_.getActReply()) {
      blocks.push_back(b);
    }
    for (auto& b : cmdReply_.getActReply()) {
      blocks.push_back(b);
    }
    return blocks;
  }

  bool terminal() const {
    return terminal_.getBuffer().item<float>() > 0;
  }

  std::vector<int64_t> getRawInstruction() const {
    return instruction_.getRawInstruction();
  }

  void writeCmds(
      const GameEnv& env,
      const CmdReceiver& receiver,
      PlayerId playerId,
      Preload* preload,
      std::map<UnitId, CmdBPtr>* assignedCmds,
      std::mt19937* rng) {
    cmdReply_.writeCmds(env, receiver, playerId, preload, assignedCmds, rng);
  }

  // AggregatedCmdReply& cmdReply() {
  //   return cmdReply_;
  // }

  bool skim(const RTSStateExtend& state, const AI& ai, bool nofow=false);

  void computeFeatures(
      Preload& preload,
      const CmdReceiver& receiver,
      const GameEnv& env,
      PlayerId playerId,
      bool respectFow);

  // append current feature to trajectory
  void pushGameFeature();

  void pushLastRAndTerminal();

  void pushActionAndPolicy();

  // update some features, such as prevAction
  void postActUpdate();

  void updatePrevCmd(const std::map<UnitId, CmdBPtr>& assignedCmds);

  // called after each act step
  void reset();

  // called before each new game
  void newGame();

  // information for map feature
  static const int mapCoordOffset;
  static const int mapVisibilityOffset;
  static const int mapTerrainOffset;
  static const int mapArmyOffset;
  static const int mapEnemyOffset;
  static const int mapResourceOffset;
  static const int mapFeatNumChannels;

  static const int countVisibilityOffset;
  static const int countArmyOffset;
  static const int countEnemyOffset;
  static const int countResourceOffset;
  static const int countFeatNumChannels;

  const int maxNumUnits_;
  const int numPrevCmds_;
  const int mapX_;
  const int mapY_;
  const int numResourceBins_;
  const int resourceBinSize_;
  const bool useMovingAvg_;
  const float movingAvgDecay_;
  const bool verbose_;

 // private:
  void computeArmyAndResourceBasic(const Preload& preload);

  void computeEnemyBasic(int tick, const Preload& preload);

  void computeArmyExtra(
      const Preload& preload,
      const CmdReceiver& receiver,
      const GameEnv& env,
      PlayerId playerId,
      bool respectFow);

  void computeUnitConsCount(const Preload& preload);

  void computeResourceBin(const Preload& preload);

  void computeMapFeature(
      const Preload& preload, const GameEnv& env, PlayerId playerId);

  void computeLastRAndTerminal(int tick, const GameEnv& env, PlayerId playerId);

  void clearFeatures();

  OnehotInstruction instruction_;
  AggregatedCmdReply cmdReply_;

  std::map<UnitId, std::vector<int64_t>> prevCmds_;

  bool gameStart_ = true;
  int townHallDiscoveredTick_ = -1;
  float prevExploredMap_ = 0.0;
  float exploredMap_ = 0.0;

  // features for coach
  FixedLengthTrajectory countFeat_;
  FixedLengthTrajectory baseCountFeat_;
  FixedLengthTrajectory consCount_;
  FixedLengthTrajectory movingEnemyCount_;

  // features for executor & coach
  FixedLengthTrajectory mapFeat_;

  FixedLengthTrajectory numArmy_;
  FixedLengthTrajectory armyType_;
  FixedLengthTrajectory armyHp_;
  FixedLengthTrajectory armyX_;
  FixedLengthTrajectory armyY_;

  FixedLengthTrajectory cCmdType_;
  FixedLengthTrajectory cCmdUnitType_;
  FixedLengthTrajectory cCmdX_;
  FixedLengthTrajectory cCmdY_;
  FixedLengthTrajectory cCmdGatherIdx_;
  FixedLengthTrajectory cCmdAttackIdx_;
  FixedLengthTrajectory pCmdType_;

  FixedLengthTrajectory numEnemy_;
  FixedLengthTrajectory enemyType_;
  FixedLengthTrajectory enemyHp_;
  FixedLengthTrajectory enemyX_;
  FixedLengthTrajectory enemyY_;

  FixedLengthTrajectory numResource_;
  FixedLengthTrajectory resourceType_;
  FixedLengthTrajectory resourceHp_;
  FixedLengthTrajectory resourceX_;
  FixedLengthTrajectory resourceY_;

  FixedLengthTrajectory resourceBin_;

  // reward & terminal
  FixedLengthTrajectory reward_;
  FixedLengthTrajectory terminal_;

  std::vector<std::reference_wrapper<FixedLengthTrajectory>> features_;
};
