// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#include "comm_ai.h"

void WebCtrl::Extract(const RTSStateExtend& s, json* game) {
  const auto id = _raw_converter.GetPlayerId();
  ExtractWithId(s, id, game);
}

void WebCtrl::ExtractWithId(const RTSStateExtend& s, int player_id, json* game) {
  const CmdReceiver& recv = s.receiver();
  const GameEnv& env = s.env();

  env.FillHeader<save2json, json>(recv, game);
  env.FillIn<save2json, json>(player_id, recv, game);

  for (const UnitId id : _raw_converter.GetAllSelectedUnits()) {
    (*game)["selected_units"].push_back(id);
  }
}

void WebCtrl::Receive(
    const RTSStateExtend& s,
    std::vector<CmdBPtr>* cmds,
    std::vector<UICmd>* ui_cmds) {
  std::string msg;
  while (queue_.try_dequeue(msg)) {
    _raw_converter.Process(s.GetTick(), s.env(), msg, cmds, ui_cmds);
  }
}

bool WebCtrl::Ready() const {
  return server_->ready();
}

bool TCPAI::act(const RTSStateExtend& s, RTSAction* action) {
  // First we send the visualization.
  json game;
  _ctrl.Extract(s, &game);
  _ctrl.Send(game.dump());

  std::vector<CmdBPtr> cmds;
  std::vector<UICmd> ui_cmds;

  _ctrl.Receive(s, &cmds, &ui_cmds);
  // Put the cmds to action, ignore all ui cmds.
  // [TODO]: Move this to elf::game_base.h.
  action->Init(getId(), GameDef::GetNumAction(), RTSAction::CMD_BASED);
  for (auto&& cmd : cmds) {
    action->cmds().emplace(make_pair(cmd->id(), std::move(cmd)));
  }
  return true;
}

bool TCPPlayerAI::act(const RTSStateExtend& s, RTSAction* action) {
  json game;
  _ctrl.Extract(s, &game);
  _ctrl.Send(game.dump());

  auto game_status = s.env().GetGameStatus();
  if (game_status == FROZEN_STATUS) {
    return false;
  }
  std::vector<CmdBPtr> cmds;
  std::vector<UICmd> ui_cmds;
  _ctrl.Receive(s, &cmds, &ui_cmds);
  action->Init(getId(), GameDef::GetNumAction(), RTSAction::CMD_BASED);
  for (auto&& cmd : cmds) {
    action->cmds().emplace(make_pair(cmd->id(), std::move(cmd)));
  }
  for (auto&& ui_cmd : ui_cmds) {
    action->ui_cmds().emplace_back(std::move(ui_cmd));
  }
  return true;
}

bool TCPCoachAI::act(const RTSStateExtend& s, RTSAction* action) {
  json game;
  _ctrl.ExtractWithId(s, _player_id, &game);
  _ctrl.Send(game.dump());

  std::vector<CmdBPtr> cmds;
  std::vector<UICmd> ui_cmds;
  _ctrl.Receive(s, &cmds, &ui_cmds);
  action->Init(getId(), GameDef::GetNumAction(), RTSAction::CMD_BASED);
  for (auto&& cmd : cmds) {
    action->cmds().emplace(make_pair(cmd->id(), std::move(cmd)));
  }
  for (auto&& ui_cmd : ui_cmds) {
    action->ui_cmds().emplace_back(std::move(ui_cmd));
  }
  return true;
}

bool TCPSpectator::act(const RTSStateExtend& s, Action* action) {
  auto tick = s.GetTick();

  while (tick >= static_cast<int>(_history_states.size())) {
    _history_states.emplace_back();
  }
  s.Save(&_history_states[tick]);

  json game;
  _ctrl.Extract(s, &game);

  const int replay_size = GetLoadedReplaySize();
  if (replay_size > 0) {
    game["replay_length"] = GetLoadedReplayLastTick();
  }

  // action->Init(_ctrl.GetId(), "spectator");
  _ctrl.Receive(s, &action->cmds, &action->ui_cmds);
  for (const UICmd& cmd : action->ui_cmds) {
    switch (cmd.cmd) {
      case UI_CYCLEPLAYER: {
        auto id = _ctrl.GetId() == INVALID ? 0 : _ctrl.GetId() + 1;
        if (id >= s.env().GetNumOfPlayers()) {
          id = INVALID;
        }
        _ctrl.setID(id);
      } break;
      case UI_SLIDEBAR: {
        // UI controls, only works if there is a spectator.
        // cout << "Receive slider bar notification " << cmd.arg2 << endl;
        const float r = cmd.arg2 / 100.0;
        const Tick new_tick = static_cast<Tick>(GetLoadedReplayLastTick() * r);
        if (new_tick < static_cast<int>(_history_states.size())) {
          // cout << "Switch back from tick = " << tick << " to new tick = " <<
          // new_tick << endl;
          tick = new_tick;
          action->new_state = _history_states[tick];
          Relocate(tick);
        }
      } break;
      default:
        break;
    }
  }

  SendReplay(tick, action, &game);

  if (tick >= _vis_after) {
    _ctrl.Send(game.dump());
  }

  return true;
}
