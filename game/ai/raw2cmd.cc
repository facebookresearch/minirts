// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#include "raw2cmd.h"
#include "engine/cmd.gen.h"
#include "engine/cmd_specific.gen.h"
#include "engine/utils.h"

/////////// RawToCmd /////////////////
CmdInput move_event(
    const Unit& u,
    char /*hotkey*/,
    const PointF& p,
    const UnitId& target_id,
    const GameEnv& env) {
  const auto& player = env.GetPlayer(u.GetPlayerId());
  if (target_id == INVALID || !player.FilterWithFOW(*env.GetUnit(target_id), true)) {
    if (!p.IsInvalid()) {
      return CmdInput(CmdInput::CI_MOVE, u.GetId(), p, INVALID);
    }
    return CmdInput();
  } else {
    const auto* target_unit = env.GetUnit(target_id);
    if (target_unit->IsResource()) {
      UnitId town_hall = env.FindClosestTownHall(u.GetPlayerId(), p);
      return CmdInput(CmdInput::CI_GATHER, u.GetId(), p, target_id, town_hall);
    } else {
      return CmdInput(CmdInput::CI_ATTACK, u.GetId(), p, target_id);
    }
  }
  return CmdInput();
}

CmdInput attack_event(
    const Unit& u,
    char /*hotkey*/,
    const PointF& p,
    const UnitId& target_id,
    const GameEnv& env) {
  // Don't need to check hotkey since there is only one type of action.
  // cout << "In attack command [" << hotkey << "] @" << p << " target: " <<
  // target_id << endl;
  if (target_id == INVALID) {
    if (!p.IsInvalid()) {
      UnitId new_target_id = env.FindClosestEnemy(u.GetPlayerId(), p, 1.5);
      if (new_target_id == INVALID) {
        return CmdInput(CmdInput::CI_MOVE, u.GetId(), p, new_target_id);
      } else {
        return CmdInput(CmdInput::CI_ATTACK, u.GetId(), p, new_target_id);
      }
    }
    return CmdInput();
  } else {
    return CmdInput(CmdInput::CI_ATTACK, u.GetId(), p, target_id);
  }
  return CmdInput();
}

CmdInput gather_event(
    const Unit& u,
    char /*hotkey*/,
    const PointF& p,
    const UnitId& target_id,
    const GameEnv& env) {
  // Don't need to check hotkey since there is only one type of action.
  // cout << "In gather command [" << hotkey << "] @" << p << " target: " <<
  // target_id << endl;
  UnitId town_hall = env.FindClosestTownHall(u.GetPlayerId(), p);
  return CmdInput(CmdInput::CI_GATHER, u.GetId(), p, target_id, town_hall);
}

CmdInput build_event(
    const Unit& u,
    char hotkey,
    const PointF& p,
    const UnitId& /*target_id*/,
    const GameEnv& env) {
  // Send the build command.
  UnitType t = u.GetUnitType();

  // don't need a target point for buildings
  PointF build_p;
  if (t == PEASANT) {
    if (p.IsInvalid()) {
      return CmdInput();
    }
    build_p = p;
  }

  UnitType build_type = env.GetGameDef().unit(t).GetUnitTypeFromHotKey(hotkey);
  if (build_type == INVALID_UNITTYPE) {
    return CmdInput();
  }
  return CmdInput(
      CmdInput::CI_BUILD, u.GetId(), build_p, INVALID, INVALID, build_type);
}

std::string read_instruction(std::istream& is) {
  std::string instruction;
  for (std::string piece; is >> piece;) {
    if (!instruction.empty()) {
      instruction += ' ';
    }
    instruction += piece;
  }
  return instruction;
}

void RawToCmd::add_hotkey(
    const std::string& s,
    EventResp f,
    bool is_build_cmd) {
  for (size_t i = 0; i < s.size(); ++i) {
    _hotkey_maps.insert(make_pair(s[i], f));
    if (is_build_cmd) {
      _build_hotkeys.insert(s[i]);
    }
  }
}

void RawToCmd::setup_hotkeys() {
  add_hotkey("a", attack_event);
  add_hotkey("~m", move_event);
  add_hotkey("g", gather_event);
  // units
  add_hotkey("percud", build_event, false);
  // buildings
  add_hotkey("hbslwt", build_event, true);
}

RawMsgStatus RawToCmd::Process(
    Tick tick,
    const GameEnv& env,
    const std::string& s,
    std::vector<CmdBPtr>* cmds,
    std::vector<UICmd>* ui_cmds) {
  // Raw command:
  //   t 'L' i j: left click at (i, j)
  //   t 'R' i j: right clock at (i, j)
  //   t 'B' x0 y0 x1 y1: bounding box at (x0, y0, x1, y1)
  //   t 'S' percent: slide bar to percentage
  //   t 'F'        : faster
  //   t 'W'        : slower
  //   t 'C'        : cycle through players.
  //   t lowercase : keyboard click.
  // t is tick.
  if (s.empty())
    return PROCESSED;
  assert(cmds != nullptr);
  assert(ui_cmds != nullptr);

  Tick t;
  char c;
  float percent;
  std::string instruction;
  PointF p, p2;
  std::set<UnitId> selected;

  const RTSMap& m = env.GetMap();

  std::istringstream ii(s);
  std::string cc;
  ii >> t >> cc;
  if (cc.size() != 1)
    return PROCESSED;
  c = cc[0];

  switch (c) {
    case 'R':
      ii >> p;
      if (!m.IsIn(p)) {
        return FAILED;
      }
      if (!env.IsGameActive()) {
        return FAILED;
      }
      {
        UnitId closest_id = m.GetClosestUnitId(p, 2.5);
        if (closest_id != INVALID && !_sel_unit_ids.empty()) {
          selected.insert(closest_id);
        }
      }
      break;
    case 'L':
      ii >> p;
      if (!m.IsIn(p)) {
        return FAILED;
      }
      if (!env.IsGameActive()) {
        return FAILED;
      }
      {
        UnitId closest_id = m.GetClosestUnitId(p, 2.5);
        if (closest_id != INVALID) {
          if (PointF::L2Sqr(_last_left_click_p, p) < 1e-4) {
            const auto& multi_selected = MultiSelect(closest_id, env);
            selected.insert(multi_selected.begin(), multi_selected.end());
          } else {
            selected.insert(closest_id);
          }
          _last_left_click_p = p;
          selected = FilterSelectedUnits(env, selected);
        }
      }
      break;
    case 'B':
      ii >> p >> p2;
      if (!m.IsIn(p) || !m.IsIn(p2)) {
        return FAILED;
      }
      // Reorder the four corners.
      if (p.x > p2.x) {
        std::swap(p.x, p2.x);
      }
      if (p.y > p2.y) {
        std::swap(p.y, p2.y);
      }
      if (!env.IsGameActive()) {
        return FAILED;
      }
      {
        selected = FilterSelectedUnits(env, m.GetUnitIdInRegion(p, p2));
      }
      break;
    case 'F':
      ui_cmds->push_back(UICmd::GetUIFaster());
      return PROCESSED;
    case 'W':
      ui_cmds->push_back(UICmd::GetUISlower());
      return PROCESSED;
    case 'C':
      ui_cmds->push_back(UICmd::GetUICyclePlayer());
      return PROCESSED;
    case 'S':
      ii >> percent;
      // cout << "Get slider bar notification " << percent << endl;
      ui_cmds->push_back(UICmd::GetUISlideBar(percent));
      return PROCESSED;
    case 'X':
      instruction = read_instruction(ii);
      // Hack to replace player_id with 0
      cmds->emplace_back(
          CmdCPtr(new CmdIssueInstruction(INVALID, 0, instruction)));
      std::cout << "coach issue: " << instruction << std::endl;
      break;
    case 'Z':
      instruction = read_instruction(ii);
      cmds->emplace_back(
          CmdCPtr(new CmdFinishInstruction(INVALID, 0, instruction)));
      break;
    case 'I':
      instruction = read_instruction(ii);
      cmds->emplace_back(
          CmdCPtr(
              new CmdInterruptInstruction(INVALID, 0, instruction)));
      break;
    case 'Q':
      cmds->emplace_back(
          CmdCPtr(
              new CmdWarnInstruction(INVALID, 0)));
      break;
    case 'A':
      instruction = read_instruction(ii);
      cmds->emplace_back(
          CmdCPtr(new CmdAcceptInstruction(INVALID, _player_id, instruction)));
      std::cout << "player accept: " << instruction << std::endl;
      break;

    case 'P':
      ui_cmds->push_back(UICmd::GetToggleGamePause());
      return PROCESSED;
    default:
      break;
  }

  if (c != 'L') {
    // reset last left click
    _last_left_click_p.SetInvalid();
  }

  if (!is_mouse_selection_motion(c) && !is_mouse_action_motion(c))
    _last_key = c;

  if (!is_mouse_action_motion(c) && _last_key == '~') {
    clear_state();
  }

  if (_hotkey_maps.find(_last_key) == _hotkey_maps.end()) {
    _last_key = '~';
  }

  // Rules:
  //   1. we cannot control other player's units.
  //   2. we cannot have friendly fire (enforced in the callback function)
  bool cmd_success = false;

  if (!_sel_unit_ids.empty() && selected.size() <= 1 && c != 'L') {
    UnitId id = (selected.empty() ? INVALID : *selected.begin());
    auto it_key = _hotkey_maps.find(_last_key);
    if (it_key != _hotkey_maps.end()) {
      EventResp f = it_key->second;
      for (auto it = _sel_unit_ids.begin(); it != _sel_unit_ids.end(); ++it) {
        if (ExtractPlayerId(*it) != _player_id) {
          continue;
        }

        // Only command our unit.
        const Unit* u = env.GetUnit(*it);

        // u has been deleted (e.g., killed)
        // We won't delete it in our selection, since the selection will
        // naturally update.
        if (u == nullptr) {
          continue;
        }

        CmdBPtr cmd = f(*u, _last_key, p, id, env).GetCmd();
        if (!cmd.get() ||
            !env.GetGameDef().unit(u->GetUnitType()).CmdAllowed(cmd->type())) {
          continue;
        }
        if (env.IsGameActive()) {
          // Command successful.
          cmds->emplace_back(std::move(cmd));
          cmd_success = true;
        }
      }
    }
  }

  if (!cmd_success) {
    if (is_mouse_selection_motion(c)) {
      if (!selected.empty()) {
        select_new_units(selected);
      } else {
        clear_state();
      }
    }
    /*if (is_mouse_action_motion(c)) {
      if (!selected.empty())
        select_new_units(selected);
    }*/
  } else {
    if (is_build_cmd(_last_key)) {
      clear_state();
    }
    _last_key = '~';
  }

  // Add selected units
  std::string action_msg = s;
  if (!_sel_unit_ids.empty()) {
    action_msg = s + " SEL " + GetAllSelectedUnitsStr();
  }
  if (env.IsGameActive()) {
    cmds->emplace_back(CmdIPtr(new CmdComment(INVALID, action_msg)));
  }

  if (t > tick) {
    return EXCEED_TICK;
  }
  return PROCESSED;
}

std::set<UnitId> RawToCmd::MultiSelect(const UnitId id, const GameEnv& env) {
  const auto* unit = env.GetUnit(id);
  std::set<UnitId> selected;
  selected.insert(id);
  for (const auto& pair : env.GetUnits()) {
    const auto& u = pair.second;
    if (u->IsBuilding()) {
      continue;
    }
    if (u->GetPlayerId() == unit->GetPlayerId() && u->GetUnitType() == unit->GetUnitType()) {
      selected.insert(u->GetId());
    }
  }
  return selected;
}

std::set<UnitId> RawToCmd::FilterSelectedUnits(
    const GameEnv& env,
    const std::set<UnitId>& sel_unit_ids) {
  std::set<UnitId> units, buildings;
  for (auto unit_id : sel_unit_ids) {
    const auto* unit = env.GetUnit(unit_id);
    // Don't select resources
    if (unit->IsResource()) {
      continue;
    }
    if (unit->IsTemporary()) {
      continue;
    }
    // Only select your units
    if (unit->GetPlayerId() != _player_id) {
      continue;
    }
    if (unit->IsUnit()) {
      units.insert(unit_id);
    }
    if (unit->IsBuilding()) {
      buildings.insert(unit_id);
    }
  }
  // Always return units if have any
  if (!units.empty()) {
    return units;
  }
  // For buildings, return only one or none
  if (buildings.size() == 1) {
    return buildings;
  }
  return std::set<UnitId>();
}
