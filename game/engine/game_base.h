// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include <iostream>
#include <memory>
#include <vector>

#include "engine/common.h"
#include "engine/game_state_ext.h"

#include "engine/lua/cpp_interface.h"
#include "engine/lua/lua_interface.h"
#include "game_MC/lua/cpp_interface.h"
#include "game_MC/lua/lua_interface.h"

#include "ai/ai.h"
#include "ai/comm_ai.h"
#include "ai/replay_loader.h"

#include "env_thread.h"

class RTSGame : public tube::EnvThread {
 public:
  RTSGame(const RTSGameOption& option)
      : option_(option)
      , state_(option) {
  }

  RTSGame(const RTSGame&) = delete;
  RTSGame& operator=(const RTSGame&) = delete;

  int addBot(std::shared_ptr<AI> ai) {
    const int id = state_.env().GetNumOfPlayers();
    if (ai.get() == nullptr) {
      std::cout << "Bot at " << id << " cannot be nullptr" << std::endl;
      return -1;
    }
    ai->setId(id);
    state_.AppendPlayer("player " + std::to_string(id));
    bots_.push_back(std::move(ai));
    return id;
  }

  void addSpectator(std::unique_ptr<Replayer>&& spectator) {
    assert(spectator_ == nullptr);
    spectator_ = std::move(spectator);
  }

  void addDefaultSpectator() {
    assert(spectator_ == nullptr);
    spectator_ = std::make_unique<TCPSpectator>("", 0, 8002);
  }

  virtual void mainLoop() override {
    reg_engine_lua_interfaces();
    reg_engine_cpp_interfaces(option_.lua_files);
    reg_lua_interfaces();
    reg_cpp_interfaces(option_.lua_files);

    int gameCount = 0;
    while (option_.num_games_per_thread <= 0 ||
           gameCount < option_.num_games_per_thread) {
      reset();
      oneGame();
      if (option_.num_games_per_thread > 0) {
        ++gameCount;
      }
    }
  }

 private:
  void reset() {
    state_.Reset();
    for (const auto& bot : bots_) {
      bot->newGame(state_);
    }
  }

  void oneGame() {
    state_.Init();
    // int step__ = 0;
    while (true) {
      // step__++;
      // std::cout << "step: " << step__ << std::endl;
      if (step() != GAME_NORMAL) {
        break;
      }
    }
    // Send message to AIs.
    act(false);
    gameEnd();
    state_.Finalize();
  }

  GameResult step() {
    state_.PreAct();
    act(true);
    GameResult res = state_.PostAct();
    state_.IncTick();
    return res;
  }

  void act(bool checkFrameSkip) {
    auto t = state_.GetTick();
    for (const auto& bot : bots_) {
      if (!checkFrameSkip || (t + 1) % bot->frameSkip() == 0) {
        RTSAction action;
        if (bot->act(state_, &action)) {
          // don't log cmd targets on uneven ticks
          state_.forward(action);
        }
      }
    }
    if (spectator_ != nullptr) {
      typename Replayer::Action action;
      if (spectator_->act(state_, &action)) {
        state_.forward(action);
      }
    }
  }

  void gameEnd() {
    for (const auto& bot : bots_) {
      bot->endGame(state_);
    }
    if (spectator_ != nullptr) {
      spectator_->endGame(state_);
    }
  }

  const RTSGameOption option_;
  RTSStateExtend state_;
  std::vector<std::shared_ptr<AI>> bots_;
  std::unique_ptr<Replayer> spectator_;
};
