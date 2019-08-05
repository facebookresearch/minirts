// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once
#include "engine/cmd_interface.h"

class CmdReceiver;

// RawMsgStatus.
custom_enum(RawMsgStatus, PROCESSED = 0, EXCEED_TICK, FAILED);

using EventResp = std::function<CmdInput(
    const Unit&,
    char hotkey,
    const PointF& p,
    const UnitId& target_id,
    const GameEnv&)>;

class RawToCmd {
 private:
  // Internal status.
  PlayerId _player_id;
  std::set<UnitId> _sel_unit_ids;
  char _last_key;

  std::map<char, EventResp> _hotkey_maps;
  std::set<char> _build_hotkeys;

  PointF _last_left_click_p;

  void select_new_units(const std::set<UnitId>& ids) {
    _sel_unit_ids = ids;
    _last_key = '~';
  }

  void clear_state() {
    _last_key = '~';
    _sel_unit_ids.clear();
  }

  void add_hotkey(const std::string& keys, EventResp func, bool is_build_cmd = false);
  void setup_hotkeys();

  static bool is_mouse_selection_motion(char c) {
    return c == 'L' || c == 'B';
  }

  static bool is_mouse_action_motion(char c) {
    return c == 'R';
  }

  bool is_build_cmd(char c) {
    return _build_hotkeys.find(c) != _build_hotkeys.end();
  }

 public:
  RawToCmd(PlayerId player_id = INVALID)
      : _player_id(player_id), _last_key('~') {
    setup_hotkeys();
  }

  RawMsgStatus Process(
      Tick tick,
      const GameEnv& env,
      const std::string& s,
      std::vector<CmdBPtr>* cmds,
      std::vector<UICmd>* ui_cmds);

  void setID(PlayerId id) {
    _player_id = id;
    clear_state();
  }

  PlayerId GetPlayerId() const {
    return _player_id;
  }

  bool IsUnitSelected() const {
    return !_sel_unit_ids.empty();
  }

  bool IsSingleUnitSelected() const {
    return _sel_unit_ids.size() == 1;
  }

  bool IsUnitSelected(const UnitId& id) const {
    return _sel_unit_ids.find(id) != _sel_unit_ids.end();
  }

  void ClearUnitSelection() {
    _sel_unit_ids.clear();
  }

  const std::set<UnitId>& GetAllSelectedUnits() const {
    return _sel_unit_ids;
  }

  const std::string GetAllSelectedUnitsStr() const {
    std::stringstream ss;
    for (auto uid : _sel_unit_ids) {
      ss << uid << " ";
    }
    return ss.str();
  }

  std::set<UnitId> FilterSelectedUnits(
      const GameEnv& env,
      const std::set<UnitId>& sel_unit_ids);

  std::set<UnitId> MultiSelect(
      const UnitId id,
      const GameEnv& env);

  void NotifyDeleted(const UnitId& id) {
    auto it = _sel_unit_ids.find(id);
    if (it != _sel_unit_ids.end()) {
      _sel_unit_ids.erase(it);
    }
  }
};
