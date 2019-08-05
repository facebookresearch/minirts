// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include <stdlib.h>
#include <random>
#include <vector>
#include <utility>
#include <cstring>

#include "data_channel.h"
#include "dispatcher.h"

#include "ai/ai.h"
#include "engine/mc_rule_actor.h"

using tube::DataBlock;
using tube::FixedLengthTrajectory;
using tube::DataChannel;
using tube::Dispatcher;

class RandomizedSingleUnitTypeAI : public AI {
 public:
  RandomizedSingleUnitTypeAI(
      const AIOption& opt,
      int threadId,
      std::shared_ptr<DataChannel> dc,
      UnitType unitType,
      bool buildTower,
      std::vector<UnitType>&& possibleTypes)
      : AI(opt, threadId),
        unitType_(unitType),
        buildTower_(buildTower),
        possibleTypes_(possibleTypes) {
    if (unitType_ == INVALID_UNITTYPE) {
      randomUnitType_ = true;
    }

    if (dc != nullptr) {
      result_ = std::make_shared<DataBlock>(
          "r", std::initializer_list<int64_t>{1}, torch::kFloat32);
      uType_ = std::make_shared<DataBlock>(
          "utype", std::initializer_list<int64_t>{1}, torch::kInt64);
      resourceScale_ = std::make_shared<DataBlock>(
          "resource_scale", std::initializer_list<int64_t>{1}, torch::kFloat32);

      // init val, assume all win
      result_->getBuffer()[0] = 1.0;

      dispatcher_ = std::make_unique<Dispatcher>(std::move(dc));
      dispatcher_->addDataBlocks({uType_, result_}, {uType_, resourceScale_});
    }
  }

  virtual void setId(int id) override {
    AI::setId(id);
    // std::cout << "set id for thread: " << threadId_ << std::endl;
    // cannot put setRandomness into constructor
    // since random seed in set in AI::setId()
    // setRandomness();
  }

  bool newGame(const RTSStateExtend&) override {
    setRandomness();
    return true;
  }

  bool endGame(const RTSStateExtend& s) override {
    if (!AI::endGame(s)) {
      return false;
    }

    if (dispatcher_ == nullptr) {
      return true;
    }

    // set reward
    auto winner_id = s.env().GetWinnerId();
    if (winner_id == getId()) {
      result_->getBuffer()[0] = 1.0;
    } else {
      result_->getBuffer()[0] = -1.0;
    }

    return true;
  }

  bool act(const RTSStateExtend& state, RTSAction* action) override;

 protected:
  void setRandomness() {
    if (dispatcher_ != nullptr) {
      // std::cout << "dispatch go" << std::endl;
      dispatcher_->dispatch();
      int64_t uType = uType_->getBuffer().item<int64_t>();
      unitType_ = possibleTypes_.at(uType);
      option_.resource_scale = resourceScale_->getBuffer().item<float>();
      // std::cout << "get unit type from channel: " << uType << std::endl;
      // std::cout << "get resource scale from channel: " << option_.resource_scale << std::endl;
    } else if (randomUnitType_) {
      unitType_ = sampleUnitType();
    }

    maxArmy_ = getRandomArmyMax();
    minArmy_ = getRandomArmyMin();
  }

  UnitType sampleUnitType() {
    std::uniform_int_distribution<int> dist(0, possibleTypes_.size() - 1);
    int typeIdx = dist(rng);
    return possibleTypes_[typeIdx];
  }

  virtual int getRandomArmyMin() = 0;

  virtual int getRandomArmyMax() = 0;

  // for win/loss , unit type, resource scale communication
  std::shared_ptr<DataBlock> result_;
  std::shared_ptr<DataBlock> uType_;
  std::shared_ptr<DataBlock> resourceScale_;

  std::unique_ptr<Dispatcher> dispatcher_;

  UnitType unitType_;
  bool buildTower_;
  std::vector<UnitType> possibleTypes_;

  bool randomUnitType_ = false;
  float lastR_ = 1;  // 1 means win, -1 means loss

  // reset every epoch
  int minArmy_ = -1;
  int maxArmy_ = -1;

  // for human play
  int inactiveUntil_ = -1;
};

class SimpleAI : public RandomizedSingleUnitTypeAI {
 public:
  SimpleAI(
      const AIOption& opt,
      int threadId,
      UnitType unitType)
      : RandomizedSingleUnitTypeAI(
            opt,
            threadId,
            nullptr,
            unitType,
            false,
            {SPEARMAN, SWORDMAN, CAVALRY, ARCHER, DRAGON}) {}

  virtual int getRandomArmyMin() override {
    return 3;
  }

  virtual int getRandomArmyMax() override {
    return 3;
  }
};

class MediumAI : public RandomizedSingleUnitTypeAI {
 public:
  MediumAI(
      const AIOption& opt,
      int threadId,
      UnitType unitType,
      bool buildTower)
      : RandomizedSingleUnitTypeAI(
            opt,
            threadId,
            nullptr,
            unitType,
            buildTower,
            {SPEARMAN, SWORDMAN, CAVALRY, ARCHER, DRAGON}) {}

  MediumAI(
      const AIOption& opt,
      int threadId,
      std::shared_ptr<DataChannel> dc,
      UnitType unitType,
      bool buildTower)
      : RandomizedSingleUnitTypeAI(
            opt,
            threadId,
            dc,
            unitType,
            buildTower,
            {SPEARMAN, SWORDMAN, CAVALRY, ARCHER, DRAGON}) {}

  virtual int getRandomArmyMin() override {
    std::uniform_int_distribution<int> dist(3, 5);
    return dist(rng);
  }

  virtual int getRandomArmyMax() override {
    std::uniform_int_distribution<int> dist(5, 7);
    return dist(rng);
  }
};

class StrongAI : public AI {
 public:
  StrongAI(const AIOption& opt, int threadId)
      : AI(opt, threadId),
        unitType_(INVALID_UNITTYPE),
        lastSeenEnemyUnit_(INVALID_UNITTYPE),
        minArmy_(-1),
        maxArmy_(-1),
        scouted_(false) {}

  // virtual std::string getSignature() const override {
  //   return std::string("strong");
  // }

  void setId(int id) override {
    AI::setId(id);
    minArmy_ = getRandomArmy(3, 5);
    maxArmy_ = getRandomArmy(5, 7);
  }

  bool endGame(const RTSStateExtend& s) override {
    if (!AI::endGame(s)) {
      return false;
    }
    minArmy_ = getRandomArmy(3, 5);
    maxArmy_ = getRandomArmy(5, 7);
    return true;
  }

 protected:
  UnitType unitType_;
  UnitType lastSeenEnemyUnit_;
  int minArmy_;
  int maxArmy_;
  bool scouted_;

  bool act(const RTSStateExtend& state, RTSAction* action) override;

  int getRandomArmy(int min, int max) {
    std::uniform_int_distribution<int> dist(min, max);
    return dist(rng);
  }
};
