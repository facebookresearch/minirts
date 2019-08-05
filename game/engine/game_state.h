// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include <thread>

#include "common.h"
#include "game_action.h"
#include "game_env.h"
#include "game_option.h"

class RTSState_ {
 public:
  RTSState_(const RTSGameOption& option);

  RTSState_(const RTSState_& s) = delete;
  RTSState_& operator=(const RTSState_& s) = delete;

  void Save(std::string* s) const {
    serializer::saver saver(true);
    _env.SaveSnapshot(saver);
    _cmd_receiver.SaveCmdReceiver(saver);
    *s = saver.get_str();
  }

  void Load(const std::string& s) {
    serializer::loader loader(true);
    loader.set_str(s);
    _env.LoadSnapshot(loader);
    _cmd_receiver.LoadCmdReceiver(loader);
  }

  const GameEnv& env() const {
    return _env;
  }

  const CmdReceiver& receiver() const {
    return _cmd_receiver;
  }

  // TODO: temp hack for executor ai
  CmdReceiver& receiver() {
    return _cmd_receiver;
  }

  Tick GetTick() const {
    return _cmd_receiver.GetTick();
  }

  std::string GetUniquePrefix(bool ignoreCounter = false) const {
    int game_counter = _env.GetGameCounter();
    std::string prefix = _option.save_replay_prefix;
    if (!ignoreCounter) {
      prefix += "_" + std::to_string(game_counter);
    }
    return prefix;
  }

  void SetGlobalStats(GlobalStats* stats) {
    _cmd_receiver.GetGameStats().SetGlobalStats(stats);
  }

  bool NeedSaveReplay(Tick t) const;

  void SaveReplay();

  bool Init();

  void PreAct();

  GameResult PostAct();

  void IncTick();

  void Finalize();

  bool Reset();

  // forward actions to cmd_receiver
  bool forward(RTSAction&);

  // void AppendPlayer();
  void AppendPlayer(
      const std::string& name);  //, PlayerType player_type=PT_PLAYER);

  // void RemoveLastPlayer();

  // TODO: temp hack to get game state in Json
  std::string GetJsonState() {
    return _json_state;
  }

  void SetJsonState(const std::string& state) {
    assert(_json_state.empty());
    _json_state = state;
  }

  void ClearJsonState() {
    _json_state.clear();
  }

 protected:
  void SleepUntilQuota() {
    if (_option.main_loop_quota > 0) {
      std::this_thread::sleep_until(
          _time_loop_start +
          std::chrono::milliseconds(_option.main_loop_quota));
    }
  }

  RTSGameOption _option;
  int _seed;
  std::mt19937 _rng;
  GameEnv _env;
  CmdReceiver _cmd_receiver;
  std::chrono::time_point<std::chrono::system_clock> _time_loop_start;
  std::string _json_state;
};
