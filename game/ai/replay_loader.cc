// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#include "engine/game_state_ext.h"
#include "engine/cmd.gen.h"
#include "engine/serializer.h"

#include "replay_loader.h"

bool ReplayLoader::Load(const std::string& replay_filename) {
  // Load the replay_file (which is a action sequence)
  // Each action looks like the following:
  //    Tick, CmdType, UnitId, UnitType, Loc
  if (replay_filename.empty()) {
    return false;
  }
  _replay_filename = replay_filename;
  _loaded_replay.clear();
  _next_replay_idx = 0;

  serializer::loader loader(false);

  if (!loader.read_from_file(replay_filename)) {
    std::cout << "Loaded replay " << replay_filename << " failed!"
              << std::endl;
    return false;
  }
  std::vector<CmdBPtr> chunk;
  while (loader.get().peek() != EOF) {
    chunk.clear();
    loader >> chunk;
    for (auto&& item : chunk) {
      _loaded_replay.push_back(std::move(item));
    }
  }
  std::cout << "Loaded replay, size = " << _loaded_replay.size() << std::endl;

  return true;
}

void  ReplayLoader::SendReplay(Tick tick, Action* actions, json* game) {
  actions->cmds.clear();
  actions->restart = (tick == 0 && !_loaded_replay.empty());

  while (_next_replay_idx < _loaded_replay.size()) {
    const auto& cmd = _loaded_replay[_next_replay_idx];
    if (cmd->tick() > tick) {
      break;
    }
    if (cmd->type() == COMMENT) {
      const auto& comment = dynamic_cast<CmdComment*>(cmd.get())->comment();
      ApplyCmdComment(tick, comment);
    } else {
      actions->cmds.emplace_back(cmd->clone());
    }
    _next_replay_idx++;
  }
  if (game != nullptr) {
    for (const auto id : _selected_units) {
      (*game)["selected_units"].push_back(id);
    }
    if (tick - _last_update < 20) {
      (*game)["mouse"] = _mouse;
    }
  }

  if (!_replay_filename.empty() && _next_replay_idx == _loaded_replay.size()) {
    actions->force_terminate = true;
  }
}

void ReplayLoader::Relocate(Tick tick) {
  // Adjust the _next_replay_idx pointer so that it points to the position just
  // before the tick.
  _next_replay_idx = 0;
  while (_next_replay_idx < _loaded_replay.size()) {
    if (_loaded_replay[_next_replay_idx]->tick() >= tick) {
      break;
    }
    _next_replay_idx++;
  }
}

void ReplayLoader::ApplyCmdComment(Tick tick, const std::string& comment) {
  // Extract mouse actions and selected units
  // Format example: 35 B 0.75 2.9 13 11.8 SEL 2 3 4
  std::stringstream ss(comment);
  std::string mouse_act;
  Tick original_tick;
  ss >> original_tick >> mouse_act;
  if (mouse_act == "B" || mouse_act == "R" || mouse_act == "L") {
    _last_update = tick;
    _mouse["act"] = mouse_act;
    float x, y;
    ss >> x >> y;
    _mouse["x1"] = x;
    _mouse["y1"] = y;
    if (mouse_act == "B") {
      ss >> x >> y;
      _mouse["x2"] = x;
      _mouse["y2"] = y;
    }
    std::string selection;
    ss >> selection;
    UnitId id;
    _selected_units.clear();
    while (ss >> id) {
      _selected_units.push_back(id);
    }
  }
}

bool Replayer::act(const RTSStateExtend& s, Action* a) {
  Tick t = s.GetTick();
  SendReplay(t, a, nullptr);
  return true;
}
