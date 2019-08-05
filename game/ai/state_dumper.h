// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include <list>
#include <nlohmann/json.hpp>

#include "engine/utils.h"
#include "engine/rule_actor.h"
#include "engine/cmd_target.h"

#include "ai/ai.h"
#include "ai/replay_loader.h"

class StateDumper : public AI {
 public:
  StateDumper(const AIOption opt, std::string replayFile)
      : AI(opt, 0),
        nextReplayIdx_(0) {
    replayLoader_.Load(replayFile);
  }

  virtual int frameSkip() const override {
    return 1;
  }

  virtual bool act(const RTSStateExtend& state, RTSAction*) override;

  virtual bool endGame(const RTSStateExtend& s) override {
    if (needLogState()) {
      std::string filename = s.GetUniquePrefix(true);
      filename += (".p" + std::to_string(getId()) + ".json");
      auto winner = s.env().GetWinnerId();
      float reward = (winner == getId() ? 1.0 : -1.0);
      gameStates_.push_back(reward);
      saveGameStates(filename);
      clearGameStates();
    }
    return AI::endGame(s);
  }

 private:
  bool needLogState() const {
    return option_.log_state;
  }

  void logState(int tick, nlohmann::json&& data) {
    data["tick"] = "tick" + std::to_string(tick);
    assert(gameStates_[tick].is_null());
    gameStates_[tick] = data;
  }

  void patchMyUnits(int tick, nlohmann::json&& data) {
    assert(!gameStates_[tick].is_null());

    for (auto new_unit = data["my_units"].begin();
         new_unit != data["my_units"].end();
         ++new_unit) {
      auto unit_id = (*new_unit)["unit_id"];
      auto old_unit = gameStates_[tick]["my_units"].begin();
      for (; old_unit != gameStates_[tick]["my_units"].end(); ++old_unit) {
        if ((*old_unit)["unit_id"] == unit_id) {
          *old_unit = *new_unit;
          break;
        }
      }
      if (old_unit == gameStates_[tick]["my_units"].end()) {
        gameStates_[tick]["my_units"].push_back(*new_unit);
      }
    }
  }

  void appendTargetJson(int tick, nlohmann::json&& target) {
    assert(needLogState());
    if (gameStates_[tick].find("targets") == gameStates_[tick].end()) {
      gameStates_[tick]["targets"] = nlohmann::json::array();
    }
    gameStates_[tick]["targets"].push_back(target);
  }

  void saveGameStates(std::string filename) {
    std::ofstream file(filename);
    // std::cout << "save to " << filename << std::endl;
    file << std::setw(4) << gameStates_ << std::endl;
  }

  void clearGameStates() {
    gameStates_.clear();
  }

  bool processInstruction(const CmdBPtr& cmd);

  ReplayLoader replayLoader_;
  size_t nextReplayIdx_;
  std::string currentInstruction_;
  std::string nextInstruction_;
  nlohmann::json gameStates_;
};
