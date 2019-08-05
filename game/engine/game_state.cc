// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#include "game_state.h"
#include "cmd_target.h"
#include "utils.h"

RTSState_::RTSState_(const RTSGameOption& option)
    : _option(option)
    , _env(option.max_num_units_per_player) {
  if (option.seed == 0) {
    _seed = get_time_microseconds_mod_by();
  } else {
    _seed = option.seed;
  }
  _rng.seed(_seed);

  _env.InitGameDef(option.lua_files);
  _env.ClearAllPlayers();
}

void RTSState_::AppendPlayer(const std::string& name) {
  _env.AddPlayer(name);
}

// void RTSState_::RemoveLastPlayer() {
//   _env.RemovePlayer();
// }

bool RTSState_::NeedSaveReplay(Tick t) const {
  // no need to save at all
  if (_option.save_replay_prefix.empty()) {
    return false;
  }
  // no need to save replay during game
  if (_option.save_replay_freq <= 0) {
    return false;
  }
  return (t % _option.save_replay_freq == 0);
}

void RTSState_::SaveReplay() {
  int game_counter = _env.GetGameCounter();
  if (_option.save_replay_per_games == 0 ||
      game_counter % _option.save_replay_per_games == 0) {
    _cmd_receiver.SaveReplay(GetUniquePrefix() + ".rep");
  }
}

bool RTSState_::Init() {
  int game_counter = _env.GetGameCounter();
  int seed;
  if (game_counter == 0) {
    seed = _seed;  // for debug
  } else {
    seed = _rng();
  }
  _cmd_receiver.SendCmdWithTick(CmdBPtr(new CmdRandomSeed(INVALID, seed)), 0);
  for (auto&& cmd_pair : _env.GetGameDef().GetInitCmds(_option, seed)) {
    _cmd_receiver.SendCmdWithTick(std::move(cmd_pair.first), cmd_pair.second);
  }
  return true;
}

void RTSState_::PreAct() {
  _time_loop_start = std::chrono::system_clock::now();
  _env.UpdateTemporaryUnits(GetTick());
}

GameResult RTSState_::PostAct() {
  _env.Forward(&_cmd_receiver);

  _cmd_receiver.ExecuteDurativeCmds(_env);
  _cmd_receiver.ExecuteImmediateCmds(&_env);
  _cmd_receiver.ExecuteControlCmds(&_env);

  _env.ComputeFOW();

  Tick t = _cmd_receiver.GetTick();

  PlayerId winner_id = _env.GetGameDef().CheckWinner(_env);
  _env.SetWinnerId(winner_id);

  // Periodically save replays
  if (NeedSaveReplay(t)) {
    SaveReplay();
  }

  // Check winning condition
  bool run_normal = _cmd_receiver.GetGameStats().CheckGameSmooth(t);
  if (winner_id != INVALID || t >= _option.max_tick || !run_normal ||
      _cmd_receiver.GetForceTerminate()) {
    _env.SetTermination();
    return run_normal ? GAME_END : GAME_ERROR;
  }

  return GAME_NORMAL;
}

void RTSState_::IncTick() {
  _cmd_receiver.IncTick();
  SleepUntilQuota();
}

void RTSState_::Finalize() {
  std::string cmt = std::to_string(GetTick()) + " ";
  PlayerId winner_id = _env.GetWinnerId();

  std::string player_name;
  if (winner_id == INVALID) {
    player_name = "failed or tie";
  } else {
    player_name = _env.GetPlayer(winner_id).GetName();
  }
  _cmd_receiver.GetGameStats().SetWinner(player_name, winner_id);

  if (_option.save_replay_prefix.empty()) {
    return;
  }

  if (winner_id == INVALID) {
    cmt += "Game Terminated";
  } else {
    cmt += "WON " + player_name + " " + std::to_string(winner_id);
  }
  _cmd_receiver.SendCmd(CmdBPtr(new CmdComment(INVALID, cmt)));
  _cmd_receiver.ExecuteImmediateCmds(&_env);

  SaveReplay();
}

bool RTSState_::Reset() {
  _cmd_receiver.ResetTick();
  _cmd_receiver.ClearCmd();
  _env.Reset();
  return true;
}

bool RTSState_::forward(RTSAction& action) {
  if (!action.Send(_env, _cmd_receiver)) {
    return false;
  }

  // Finally send these commands.
  for (auto it = action.cmds().begin(); it != action.cmds().end(); ++it) {
    if (it->second->type() == ISSUE_INSTRUCTION) {
      const auto* cmd = dynamic_cast<CmdIssueInstruction*>(it->second.get());
      CmdBPtr cmd_issue = std::make_unique<CmdIssueInstruction>(
          INVALID, cmd->player_id(), cmd->instruction(), cmd->change_state());

      _cmd_receiver.SendCmd(std::move(cmd_issue));
      continue;
    }

    if (it->second->type() == FINISH_INSTRUCTION ||
        it->second->type() == ACCEPT_INSTRUCTION ||
        it->second->type() == COMMENT) {
      _cmd_receiver.SendCmd(std::move(it->second));
      continue;
    }

    if (it->second->type() == INTERRUPT_INSTRUCTION) {
      const auto& instruction =
          dynamic_cast<CmdInterruptInstruction*>(it->second.get())
              ->instruction();
      _cmd_receiver.SendCmd(
          CmdBPtr(new CmdInterruptInstruction(INVALID, 0, instruction)));
      continue;
    }

    if (it->second->type() == WARN_INSTRUCTION) {
      _cmd_receiver.SendCmd(CmdBPtr(new CmdWarnInstruction(INVALID, 0)));
      continue;
    }

    // filter out duplicated attack
    if (IsDuplicatedAttack(it->second, it->first, _cmd_receiver)) {
      continue;
    }

    if (IsDuplicatedGather(it->second, it->first, _cmd_receiver)) {
      continue;
    }

    // filter out attack on temp buildings
    if (IsAttackTemporaryBuilding(it->second, _env)) {
      continue;
    }

    // sanity checks
    const Unit* u = _env.GetUnit(it->first);
    auto gamedef = _env.GetGameDef();

    if (u == nullptr) {
      std::cout << "Error: cannot find unit id: " << it->first << std::endl;
      assert(false);
    }
    if (u->GetPlayerId() != action.player_id()) {
      std::cout << "Error: player " << action.player_id()
                << " cannot give cmd to unit: " << it->first << std::endl;
      assert(false);
    }
    if (!gamedef.unit(u->GetUnitType()).CmdAllowed(it->second->type())) {
      std::cout << "Error: cmd not allowed:"
                << "\t unit id: " << it->first << std::endl;
      std::cout << "\t unit type: " << u->GetUnitType() << std::endl;
      std::cout << "\t cmd type: " << it->second->type() << std::endl;
      std::cout << "\t cmd info: " << it->second->PrintInfo() << std::endl;
      assert(false);
    }

    // if the building is build something, cancel new build cmd
    if (gamedef.IsUnitTypeBuilding(u->GetUnitType()) &&
        _cmd_receiver.GetUnitDurativeCmd(u->GetId()) != nullptr) {
      continue;
    }

    it->second->set_id(it->first);
    _cmd_receiver.SendCmd(std::move(it->second));
  }

  return true;
}
