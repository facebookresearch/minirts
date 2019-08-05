// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

// #include "ai.h"
#include "data_block.h"

#include "engine/utils.h"
#include "engine/cmd_target.h"
#include "engine/preload.h"

using tube::DataBlock;
using tube::FixedLengthTrajectory;

// store and process all cmds for units at one tick
class AggregatedCmdReply {
 public:
  AggregatedCmdReply(int maxNumUnit,
                     int numCmdType,
                     int numUnitType,
                     int mapX,
                     int mapY,
                     bool verbose)
      : maxNumUnit(maxNumUnit),
        numCmdType(numCmdType),
        numUnitType(numUnitType),
        mapX(mapX),
        mapY(mapY),
        verbose(verbose),
        // storage for sampled cmds
        cmdType_(maxNumUnit, 0),
        targetIdx_(maxNumUnit, 0),
        targetType_(maxNumUnit, 0),
        targetX_(maxNumUnit, 0),
        targetY_(maxNumUnit, 0) {
    // buffers for data transfer
    numUnit_ = std::make_shared<DataBlock>(
        "num_unit", std::initializer_list<int64_t>{1}, torch::kInt64);
    contProb_ = std::make_shared<DataBlock>(
        "glob_cont_prob",
        std::initializer_list<int64_t>{2},
        torch::kFloat32);
    cmdTypeProb_ = std::make_shared<DataBlock>(
        "cmd_type_prob",
        std::initializer_list<int64_t>{maxNumUnit, numCmdType},
        torch::kFloat32);
    gatherIdxProb_ = std::make_shared<DataBlock>(
        "gather_idx_prob",
        std::initializer_list<int64_t>{maxNumUnit, maxNumUnit},
        torch::kFloat32);
    attackIdxProb_ = std::make_shared<DataBlock>(
        "attack_idx_prob",
        std::initializer_list<int64_t>{maxNumUnit, maxNumUnit},
        torch::kFloat32);
    unitTypeProb_ = std::make_shared<DataBlock>(
      "unit_type_prob",
      std::initializer_list<int64_t>{maxNumUnit, numUnitType},
      torch::kFloat32);
    buildingTypeProb_ = std::make_shared<DataBlock>(
        "building_type_prob",
        std::initializer_list<int64_t>{maxNumUnit, numUnitType},
        torch::kFloat32);
    buildingLocProb_ = std::make_shared<DataBlock>(
        "building_loc_prob",
        std::initializer_list<int64_t>{maxNumUnit, mapY * mapX},
        torch::kFloat32);
    moveLocProb_ = std::make_shared<DataBlock>(
        "move_loc_prob",
        std::initializer_list<int64_t>{maxNumUnit, mapY * mapX},
        torch::kFloat32);

    blocks_ = {
      numUnit_,
      contProb_,
      cmdTypeProb_,
      attackIdxProb_,
      gatherIdxProb_,
      unitTypeProb_,
      buildingTypeProb_,
      buildingLocProb_,
      moveLocProb_
    };
  }

  std::vector<std::shared_ptr<DataBlock>> getActReply() const {
    return blocks_;
  }

  void logCmds(const Preload& preload, const GameEnv& env) const;

  bool sampleCmds(std::mt19937* rng);

  void writeCmds(
      const GameEnv& env,
      const CmdReceiver& receiver,
      PlayerId playerId,
      Preload* preload,
      std::map<UnitId, CmdBPtr>* assignedCmds,
      std::mt19937* rng);

  void reset() {
    for (auto& b : blocks_) {
      b->getBuffer().zero_();
    }
    clearCmds();
  }

  void print() {
    std::cout << "Num Units: " << numUnit_ << std::endl;
    std::cout << "replying cmd type " << std::endl;
    for (int i = 0; i < (int)cmdType_.size(); ++i) {
      std::cout << cmdType_[i] << ", ";
    }
    std::cout << std::endl;
  }

  const int maxNumUnit;  // should be == that in python side
  const int numCmdType;
  const int numUnitType;
  const int mapX;
  const int mapY;
  const int verbose;

 protected:
  int numUnit() const {
    return (int)numUnit_->getBuffer().item<int64_t>();
  }

  void clearCmds() {
    std::fill(cmdType_.begin(), cmdType_.end(), 0);
    std::fill(targetIdx_.begin(), targetIdx_.end(), 0);
    std::fill(targetType_.begin(), targetType_.end(), 0);
    std::fill(targetX_.begin(), targetX_.end(), 0);
    std::fill(targetY_.begin(), targetY_.end(), 0);
  }

  // only act, not used for training
  std::shared_ptr<DataBlock> numUnit_;
  std::shared_ptr<DataBlock> contProb_;
  std::shared_ptr<DataBlock> cmdTypeProb_;
  std::shared_ptr<DataBlock> attackIdxProb_;
  std::shared_ptr<DataBlock> gatherIdxProb_;
  std::shared_ptr<DataBlock> unitTypeProb_;  // for build unit
  std::shared_ptr<DataBlock> buildingTypeProb_; // for build building
  std::shared_ptr<DataBlock> buildingLocProb_;
  std::shared_ptr<DataBlock> moveLocProb_;

  // container for all blocks
  std::vector<std::shared_ptr<DataBlock>> blocks_;

  // used to store sampled cmds
  std::vector<int64_t> cmdType_;
  std::vector<int64_t> targetIdx_;
  std::vector<int64_t> targetType_;
  std::vector<int64_t> targetX_;
  std::vector<int64_t> targetY_;
};
