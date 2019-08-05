// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include "cmd.h"
#include "game_stats.h"
#include "ui_cmd.h"

#include "pq_extend.h"
#include <map>

class GameEnv;

// Receive command and record them in the history.
// The cmds are ordered so that the lowest level is executed first.
class CmdReceiver {
 private:
  Tick _tick;
  int _cmd_next_id;
  bool _reset_file;

  GameStats _stats;

  p_queue<CmdIPtr> _immediate_cmd_queue;
  p_queue<CmdCPtr> _control_cmd_queue;
  p_queue<CmdDPtr> _durative_cmd_queue;

  std::vector<CmdBPtr> _cmd_history;
  std::vector<CmdBPtr> _cmd_full_history;

  // Record current state of each unit. Note that this pointer does not own
  // anything. When the command is destroyed, we should manually delete the
  // entry as well.
  std::map<UnitId, CmdDurative*> _unit_durative_cmd;

  // Whether we save the current issued command to the history buffer.
  bool _save_to_history;

  bool _force_terminate = false;

  bool _path_planning_verbose = false;

 public:
  CmdReceiver()
      : _tick(0)
      , _cmd_next_id(0)
      , _reset_file(true)
      , _save_to_history(true) {
  }

  void SetForceTerminate() {
    _force_terminate = true;
  }

  bool GetForceTerminate() const {
    return _force_terminate;
  }

  const GameStats& GetGameStats() const {
    return _stats;
  }
  GameStats& GetGameStats() {
    return _stats;
  }

  Tick GetTick() const {
    return _tick;
  }
  Tick GetNextTick() const {
    return _tick + 1;
  }
  inline void IncTick() {
    _tick++;
    _stats.IncTick();
  }
  inline void ResetTick() {
    _tick = 0;
    _stats.Reset();
  }

  void SetPathPlanningVerbose(bool verbose) {
    _path_planning_verbose = verbose;
  }

  bool GetPathPlanningVerbose() const {
    return _path_planning_verbose;
  }

  const std::vector<CmdBPtr>& GetHistory() const {
    return _cmd_full_history;
  }

  void ClearCmd() {
    while (!_immediate_cmd_queue.empty()) {
      _immediate_cmd_queue.pop();
    }
    while (!_control_cmd_queue.empty()) {
      _control_cmd_queue.pop();
    }
    while (!_durative_cmd_queue.empty()) {
      _durative_cmd_queue.pop();
    }
    _cmd_history.clear();
    _cmd_full_history.clear();
    _unit_durative_cmd.clear();
    _cmd_next_id = 0;
    _reset_file = true;
  }

  // Send command with the current tick.
  bool SendCmd(CmdBPtr&& cmd);

  // Send command with a specific tick.
  bool SendCmdWithTick(CmdBPtr&& cmd, Tick tick);

  // Set this to be true to prevent any command to be recorded in the history.
  void SetSaveToHistory(bool v) {
    _save_to_history = v;
  }

  bool IsSaveToHistory() const {
    return _save_to_history;
  }

  void PrintUnitDurativeCmd() const;

  // Start a durative cmd specified by the pointer.
  bool StartDurativeCmd(const GameEnv&, CmdDurative*);

  // Finish the durative command for unit id.
  bool CleanupDurativeCmd(const GameEnv& env, UnitId id);

  bool FinishDurativeCmd(UnitId id);

  bool FinishDurativeCmdIfDone(UnitId id);

  const CmdDurative* GetUnitDurativeCmd(UnitId id) const;

  // Save replay to a file.
  // Important: it clears the actions list.
  bool SaveReplay(const std::string& replay_filename);

  // Execute Durative Commands. This will not change the game environment.
  void ExecuteDurativeCmds(const GameEnv& env);

  // Execute Immediate Commands. This will change the game environment.
  void ExecuteImmediateCmds(GameEnv* env);

  // Execute Control Immediate Commands. This will change the game environment.
  void ExecuteControlCmds(GameEnv* env);

  // CmdReceiver has its specialized Save and Load function.
  // No SERIALIZER(...) is needed.
  void SaveCmdReceiver(serializer::saver& saver) const;

  void LoadCmdReceiver(serializer::loader& loader);

  ~CmdReceiver() {
  }
};
