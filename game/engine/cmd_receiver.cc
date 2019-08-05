// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#include "cmd_receiver.h"
#include "cmd.h"
#include "game_env.h"
#include <initializer_list>

#include "cmd.gen.h"
#include "cmd_specific.gen.h"

bool CmdReceiver::StartDurativeCmd(const GameEnv& env, CmdDurative* cmd) {
  UnitId id = cmd->id();
  if (id == INVALID)
    return false;

  CleanupDurativeCmd(env, id);
  FinishDurativeCmd(id);
  _unit_durative_cmd[id] = cmd;
  return true;
}

void CmdReceiver::PrintUnitDurativeCmd() const {
  for (auto&& cmd : _unit_durative_cmd) {
    std::cout << cmd.first << "->" << cmd.second->PrintInfo() << std::endl;
  }
}

bool CmdReceiver::CleanupDurativeCmd(const GameEnv& env, UnitId id) {
  auto it = _unit_durative_cmd.find(id);
  if (it != _unit_durative_cmd.end()) {
    it->second->Cleanup(env, this);
    return true;
  } else {
    return false;
  }
}

bool CmdReceiver::FinishDurativeCmd(UnitId id) {
  auto it = _unit_durative_cmd.find(id);
  if (it != _unit_durative_cmd.end()) {
    it->second->SetDone();
    _unit_durative_cmd.erase(it);
    return true;
  } else {
    return false;
  }
}

bool CmdReceiver::FinishDurativeCmdIfDone(UnitId id) {
  auto it = _unit_durative_cmd.find(id);
  if (it != _unit_durative_cmd.end() && it->second->IsDone()) {
    _unit_durative_cmd.erase(it);
    return true;
  } else
    return false;
}

bool CmdReceiver::SendCmd(CmdBPtr&& cmd) {
  return SendCmdWithTick(std::move(cmd), _tick);
}

bool CmdReceiver::SendCmdWithTick(CmdBPtr&& cmd, Tick tick) {
  // This id is just used to achieve a total ordering.
  if (cmd.get() == nullptr) {
    throw std::range_error("Error input cmd is nullptr!");
  }
  cmd->set_cmd_id(_cmd_next_id);
  _cmd_next_id++;
  cmd->set_tick_and_start_tick(tick);

  // Check wehther we need to save stuff to _cmd_history.  For all
  // commands that issued in ExecuteCmd(), we don't need to send them
  // to _cmd_history.
  if (IsSaveToHistory()) {
    _cmd_history.push_back(cmd->clone());
    _cmd_full_history.push_back(cmd->clone());
  }

  // Put the command to different queue
  CmdDurative* durative = dynamic_cast<CmdDurative*>(cmd.get());
  if (durative != nullptr) {
    // std::cout << "Receive Durative Cmd " << cmd->PrintInfo() << std::endl;
    cmd.release();
    _durative_cmd_queue.push(CmdDPtr(durative));
    return true;
  }

  CmdImmediate* immediate = dynamic_cast<CmdImmediate*>(cmd.get());
  if (immediate != nullptr) {
    // std::cout << "Receive Immediate Cmd " << cmd->PrintInfo() << std::endl;
    cmd.release();
    _immediate_cmd_queue.push(CmdIPtr(immediate));
    return true;
  }

  CmdControl* control = dynamic_cast<CmdControl*>(cmd.get());
  if (control != nullptr) {
    // std::cout << "Receive Control Cmd " << cmd->PrintInfo() << std::endl;
    cmd.release();
    _control_cmd_queue.push(CmdCPtr(control));
    return true;
  }

  throw std::range_error(
      "Error! the command is neither durative or immediate! " +
      cmd->PrintInfo());
  return false;
}

const CmdDurative* CmdReceiver::GetUnitDurativeCmd(UnitId id) const {
  auto it = _unit_durative_cmd.find(id);
  if (it == _unit_durative_cmd.end()) {
    return nullptr;
  } else {
    return it->second;
  }
}

bool CmdReceiver::SaveReplay(const std::string& replay_filename) {
  // Load the replay_file (which is a action sequence)
  // Each action looks like the following:
  //    Tick, CmdType, UnitId, UnitType, Loc
  if (_cmd_history.empty()) {
    return true;
  }
  serializer::saver saver(false);
  saver << _cmd_history;
  if (_reset_file) {
    if (!saver.write_to_file(replay_filename)) {
      return false;
    }
    _reset_file = false;
  } else {
    if (!saver.append_to_file(replay_filename)) {
      return false;
    }
  }
  _cmd_history.clear();

  return true;
}

void CmdReceiver::ExecuteDurativeCmds(const GameEnv& env) {
  // Don't run any commands when game is not active
  if (!env.IsGameActive()) {
    return;
  }

  SetSaveToHistory(false);

  // Execute durative cmds.
  while (!_durative_cmd_queue.empty()) {
    const CmdDPtr& cmd_ref = _durative_cmd_queue.top();
    // cout << "Top: " << cmd_ref->PrintInfo() << endl;
    if (cmd_ref->tick() > _tick)
      break;

    // If the command is done (often set by other preemptive commands, we skip.
    if (cmd_ref->IsDone()) {
      FinishDurativeCmdIfDone(cmd_ref->id());
      _durative_cmd_queue.pop();
      continue;
    }

    // Now finally we deal with the command.
    CmdDPtr cmd = _durative_cmd_queue.pop_top();
    cmd->Run(env, this);

    // If the command is not yet done, push it back to the queue.
    if (!cmd->IsDone()) {
      _durative_cmd_queue.push(std::move(cmd));
    } else {
      FinishDurativeCmd(cmd->id());
    }
  }

  // cout << "Ending ExecutiveDurativeCmds[" << _tick << "]" << endl;
  SetSaveToHistory(true);
}

void CmdReceiver::ExecuteImmediateCmds(GameEnv* env) {
  // Don't run any commands when game is not active
  if (!env->IsGameActive()) {
    return;
  }

  SetSaveToHistory(false);

  // cout << "Starting ExecutiveImmediateCmds[" << _tick << "]" << endl;

  // Execute immediate cmds, which will change the game state.
  while (!_immediate_cmd_queue.empty()) {
    const CmdIPtr& cmd_ref = _immediate_cmd_queue.top();
    // cout << "Top: " << cmd_ref->PrintInfo() << endl;
    if (cmd_ref->tick() > _tick)
      break;

    CmdIPtr cmd = _immediate_cmd_queue.pop_top();
    cmd->Run(env, this);
  }

  // cout << "Ending ExecutiveImmediateCmds[" << _tick << "]" << endl;
  SetSaveToHistory(true);
}

void CmdReceiver::ExecuteControlCmds(GameEnv* env) {
  SetSaveToHistory(false);

  // cout << "Starting ExecutiveControlCmds[" << _tick << "]" << endl;

  // Execute immediate cmds, which will change the game state.
  while (!_control_cmd_queue.empty()) {
    const CmdCPtr& cmd_ref = _control_cmd_queue.top();
    // cout << "Top: " << cmd_ref->PrintInfo() << endl;
    if (cmd_ref->tick() > _tick)
      break;

    CmdCPtr cmd = _control_cmd_queue.pop_top();
    cmd->Run(env, this);
  }

  // cout << "Ending ExecutiveControlCmds[" << _tick << "]" << endl;
  SetSaveToHistory(true);
}

void CmdReceiver::SaveCmdReceiver(serializer::saver& saver) const {
  // Do not save/load _loaded_replay, as well as command history.
  saver << _tick << _immediate_cmd_queue << _durative_cmd_queue
        << _control_cmd_queue;
}

void CmdReceiver::LoadCmdReceiver(serializer::loader& loader) {
  loader >> _tick >> _immediate_cmd_queue >> _durative_cmd_queue >>
      _control_cmd_queue;

  // Set the failed_moves.
  _stats.SetTick(_tick);

  // load durative cmd queue. Note that the priority queue is opaque
  // so we need some hacks.
  int size = _durative_cmd_queue.size();
  const CmdDPtr* c = &(_durative_cmd_queue.top());
  _unit_durative_cmd.clear();
  for (int i = 0; i < size; ++i) {
    const CmdDPtr& curr = c[i];
    if (!curr->IsDone()) {
      _unit_durative_cmd.insert(std::make_pair(curr->id(), curr.get()));
    }
  }
}
