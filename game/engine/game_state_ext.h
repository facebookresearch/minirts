// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include <algorithm>
#include <iostream>
#include <queue>
#include <set>
#include <string>

// #include "elf/ai/ai.h"
// #include "elf/utils/utils.h"
#include "engine/game_option.h"
#include "engine/game_state.h"

#include "ai/replay_loader.h"

// Extend RTSState for game play and visualization
// this replaces RTSState where visualization is needed
class RTSStateExtend : public RTSState_ {
 public:
  // Initialize the game.
  RTSStateExtend(const RTSGameOption& option)
      : RTSState_(option) {
    _paused = false;
    _env.SetTeamPlay(option.team_play);
  }

  GameResult PostAct() {
    if (_paused) {
      return GAME_NORMAL;
    }
    return RTSState_::PostAct();
  }

  void IncTick() {
    if (!_paused && _env.IsGameActive()) {
      RTSState_::IncTick();
    } else {
      SleepUntilQuota();
    }
  }

  bool forward(RTSAction& actions) {
    if (!RTSState_::forward(actions)) {
      return false;
    }
    dispatch_ui_cmds(actions.ui_cmds());
    return true;
  }

  bool forward(ReplayLoader::Action& actions) {
    if (actions.restart) {
      _cmd_receiver.ClearCmd();
    }
    if (!actions.new_state.empty()) {
      Load(actions.new_state);
    }
    for (auto&& cmd : actions.cmds) {
      // sanity checks, TODO: duplicated with base class forward
      const Unit* u = _env.GetUnit(cmd->id());
      if (cmd->id() != -1 && u == nullptr) {
        std::cout << "Tick << " << GetTick() << std::endl;
        std::cout << "Error: cannot find unit id: " << cmd->id() << std::endl;
        assert(false);
      }
      auto gamedef = _env.GetGameDef();
      if (u != nullptr &&
          !gamedef.unit(u->GetUnitType()).CmdAllowed(cmd->type())) {
        std::cout << "Error: cmd not allowed:"
                  << "\t unit id: " << cmd->PrintInfo() << std::endl;
        assert(false);
      }
      // if the building is build something, cancel new build cmd
      if (u != nullptr && gamedef.IsUnitTypeBuilding(u->GetUnitType()) &&
          _cmd_receiver.GetUnitDurativeCmd(u->GetId()) != nullptr) {
        std::cout << "Wrong build" << std::endl;
        continue;
      }

      _cmd_receiver.SendCmd(std::move(cmd));
    }
    dispatch_ui_cmds(actions.ui_cmds);

    if (actions.force_terminate) {
      std::cout << "force terminate: " << actions.force_terminate << std::endl;
      _cmd_receiver.SetForceTerminate();
    }
    return true;
  }

 private:
  // Dispatch commands received from gui.
  void dispatch_ui_cmds(const std::vector<UICmd>& ui_cmds) {
    for (const auto& cmd : ui_cmds) {
      // already handled by Spectator.
      if (cmd.cmd == UI_SLIDEBAR || cmd.cmd == UI_CYCLEPLAYER) {
        continue;
      }

      // speed control
      if (cmd.cmd == UI_FASTER_SIMULATION) {
        change_simulation_speed(1.25);
      } else if (cmd.cmd == UI_SLOWER_SIMULATION) {
        change_simulation_speed(0.8);
      } else if (cmd.cmd == TOGGLE_GAME_PAUSE) {
        _paused = !_paused;
      } else {
        std::cout << "Error: Cmd not handled! " << cmd.PrintInfo() << std::endl;
        assert(false);
      }
    }
  }

  bool change_simulation_speed(float fraction) {
    _option.main_loop_quota /= fraction;
    return true;
  }

  bool _paused;
};
