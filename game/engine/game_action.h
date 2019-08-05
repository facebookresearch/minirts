// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "cmd.h"
#include "cmd_receiver.h"
#include "game_env.h"
#include "ui_cmd.h"
#include "utils.h"

#include "cmd.gen.h"
#include "cmd_specific.gen.h"

class RTSAction {
 public:
  enum ActionType {
    CMD_BASED,
    RULE_BASED,
    INSTRUCTION_BASED,
  };

  void Init(PlayerId id, int num_action, ActionType type) {
    _player_id = id;
    _type = type;
    _num_action = num_action;
    _focused_town_hall = 0;
  }

  std::map<UnitId, CmdBPtr>& cmds() {
    return _cmds;
  }

  std::vector<UICmd>& ui_cmds() {
    return _ui_cmds;
  }

  PlayerId player_id() const {
    return _player_id;
  }

  void SetFocusedTownHall(int idx) {
    assert(idx >= 0 && idx < GameDef::GetMaxNumTownHall());
    _focused_town_hall = idx;
  }

  int GetFocusedTownHall() {
    return _focused_town_hall;
  }

  void SetAction(std::vector<int64_t>&& action) {
    _action = std::move(action);
  }

  void SetInstruction(std::vector<int64_t>&& action) {
    // std::cout << "setting action" << std::endl;
    // std::cout << "instruction: " << actions2str(action) << std::endl;
    _cmds[INVALID] = CmdCPtr(new CmdIssueInstruction(
        INVALID, _player_id, actions2str(action), false));
  }

  const std::vector<int64_t>& GetAction() const {
    assert(_type == ActionType::RULE_BASED);
    return _action;
  }

  std::string Info() {
    std::stringstream ss;
    ss << "RTSAction: ";
    for (auto a : _action) {
      ss << " " << a;
    }
    return ss.str();
  }

  bool Send(const GameEnv& env, CmdReceiver& receiver);

 protected:
  std::string actions2str(const std::vector<int64_t>& inst_char) {
    std::ostringstream oss;
    for (size_t i = 0; i < inst_char.size(); ++i) {
      if (inst_char[i] == -1) {
        break;
      }
      oss << char(inst_char[i]);
    }
    return oss.str();
  }

  PlayerId _player_id;

  ActionType _type;
  int _num_action;

  // which town_hall to focus on
  int _focused_town_hall;

  std::vector<int64_t> _action;

  std::map<UnitId, CmdBPtr> _cmds;
  std::vector<UICmd> _ui_cmds;
};
