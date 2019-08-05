// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

// #include <atomic>
// #include <algorithm>
#include "engine/ui_cmd.h"
#include "ai/save2json.h"

class RTSStateExtend;

class ReplayLoader {
 public:
  struct Action {
    std::vector<CmdBPtr> cmds;
    std::vector<UICmd> ui_cmds;
    bool restart = false;
    std::string new_state;
    bool force_terminate = false;
  };

  bool Load(const std::string& filename);
  void SendReplay(Tick tick, Action* actions, json* game);
  void Relocate(Tick tick);

  const std::vector<CmdBPtr>& GetLoadedReplay() const {
    return _loaded_replay;
  }

  int GetLoadedReplaySize() const {
    return _loaded_replay.size();
  }

  int GetLoadedReplayLastTick() const {
    return _loaded_replay.back()->tick();
  }

 private:
  void ApplyCmdComment(Tick tick, const std::string& comment);
  // Idx for the next replay to send to the queue.
  std::string _replay_filename;
  unsigned int _next_replay_idx = 0;
  std::vector<CmdBPtr> _loaded_replay;
  std::vector<UnitId> _selected_units;
  json _mouse;
  Tick _last_update;
};

class Replayer : public ReplayLoader {
 public:
  using Action = typename ReplayLoader::Action;

  Replayer(const std::string& filename) {
    Load(filename);
  }

  virtual bool act(const RTSStateExtend& s, Action* a);

  bool endGame(const RTSStateExtend&) {
    return true;
  }

  virtual ~Replayer() {}
};
