// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#include "preload.h"
#include "rule_actor.h"
#include "utils.h"

void Preload::collect_stats(const GameEnv& env,
                            int player_id,
                            const CmdReceiver& receiver,
                            bool respect_fow) {
  // Clear all data
  _my_troops.clear();
  _enemy_troops.clear();
  _enemy_all_units.clear();
  _enemy_base_targets.clear();
  _enemy_defend_targets.clear();
  _enemy_in_range_targets.clear();
  _my_army.clear();
  _cnt_under_construction.clear();
  _town_halls.clear();
  _prices.clear();

  // Initialize to a given size.
  _num_unit_type = env.GetGameDef().GetNumUnitType();
  _my_troops.resize(_num_unit_type);
  _enemy_troops.resize(_num_unit_type);
  _cnt_under_construction.resize(_num_unit_type, 0);
  _prices.resize(_num_unit_type, 0);
  _result = NOT_READY;

  // set price
  for (int i = 0; i < _num_unit_type; ++i) {
    _prices[i] = env.GetGameDef().unit((UnitType)i).GetUnitCost();
  }

  // Collect ...
  const Player& player = env.GetPlayer(player_id);
  GameEnvAspect aspect(env, player_id, respect_fow);
  UnitIterator unit_iter(aspect, UnitIterator::ALL);

  while (!unit_iter.end()) {
    // visibility is checked by unit_iter
    const Unit& u = *unit_iter;
    UnitType unit_type = u.GetUnitType();

    // Unit is still under construction (only building)
    if (u.IsTemporary()) {
      ++unit_iter;
      continue;
    }

    // if unit is resource, ignore the ownership
    if (unit_type == RESOURCE) {
      _my_troops[unit_type].push_back(&u);
      ++unit_iter;
      continue;
    }

    if (u.GetPlayerId() == player_id) {
      _my_troops[unit_type].push_back(&u);

      if (GameDef::IsUnitTypeOffensive(unit_type)) {
        _my_army.push_back(&u);
      }

      if (unit_type == TOWN_HALL) {
        _town_halls.emplace_back(&u, _num_unit_type);
      }

      // this unit is building something
      if (InCmd(receiver, u, BUILD)) {
        const CmdDurative* cmd = receiver.GetUnitDurativeCmd(u.GetId());
        assert(cmd != nullptr);
        const CmdBuild* cmd_build = dynamic_cast<const CmdBuild*>(cmd);
        assert(cmd_build != nullptr);
        UnitType ut = cmd_build->build_type();
        _cnt_under_construction[ut]++;
      }
    } else {
      // visible enemy's troop, and not resource
      _enemy_troops[unit_type].push_back(&u);

      if (unit_type != GUARD_TOWER && GameDef::IsUnitTypeBuilding(unit_type)) {
        _enemy_base_targets.push_back(&u);
      } else {
        _enemy_all_units.push_back(&u);
      }

      if (player.FilterWithFOW(u, false)) {
        _enemy_in_range_targets.push_back(&u);
      }
    }
    ++unit_iter;
  }
}

UnitId Preload::add_to_closest_town_hall(const Unit* unit) {
  float closest_sqr = 1e10;
  int closest_idx = -1;
  for (int i = 0; i < (int)_town_halls.size(); ++i) {
    float l2_sqr = PointF::L2Sqr(unit->GetPointF(), _town_halls[i].GetPointF());
    if (l2_sqr < closest_sqr) {
      closest_sqr = l2_sqr;
      closest_idx = i;
    }
  }
  assert(closest_idx >= 0);
  _town_halls[closest_idx].AddUnit(unit);
  return _town_halls[closest_idx].GetId();
}

bool Preload::maybe_add_to_defend_targets(const Unit* u) {
  float dsqr_bound = 25.0;
  for (auto unit_type : {TOWN_HALL, GUARD_TOWER}) {
    for (const Unit* defend_unit : _my_troops[unit_type]) {
      float dsqr = PointF::L2Sqr(u->GetPointF(), defend_unit->GetPointF());
      if (dsqr < dsqr_bound) {
        _enemy_defend_targets.push_back(u);
        return true;
      }
    }
  }
  return false;
}

bool Preload::maybe_add_to_gather_town_hall(const Unit* u,
                                            const CmdReceiver& receiver) {
  auto cmd = receiver.GetUnitDurativeCmd(u->GetId());
  if (cmd == nullptr || cmd->type() != GATHER) {
    return false;
  }
  auto cmd_gather = dynamic_cast<const CmdGather*>(cmd);
  auto town_hall_id = cmd_gather->town_hall();
  for (TownHall& town_hall : _town_halls) {
    if (town_hall.GetId() == town_hall_id) {
      town_hall.AddPeasant();
      return true;
    }
  }
  std::cout << "ERROR: gathering to an invalid town hall" << std::endl;
  assert(false);
  return false;
}

void Preload::GatherInfo(const GameEnv& env,
                         int player_id,
                         const CmdReceiver& receiver,
                         const std::list<UnitType>& build_queue,
                         bool respect_fow) {
  assert(player_id >= 0 && player_id < env.GetNumOfPlayers());
  collect_stats(env, player_id, receiver, respect_fow);
  const Player& player = env.GetPlayer(player_id);
  _resource = player.GetResource();
  _budget = _resource;

  // if (!env.GetGameDef().HasBase()) {
  //   return;
  // }

  if (_my_troops[TOWN_HALL].empty()) {
    _result = NO_TOWN_HALL;
    return;
  }

  // add unit in build_queue to cnt_under_construction
  for (UnitType ut : build_queue) {
    _cnt_under_construction[ut]++;
  }

  // distribute resource to closest town_hall
  for (const auto& unit : _my_troops[RESOURCE]) {
    assert(unit != nullptr);
    add_to_closest_town_hall(unit);
  }

  // distribute peasant to town_hall according to gather destination
  // TODO: will trigger error if target townhall is destroyed
  // for (const auto& unit : _my_troops[PEASANT]) {
  //   assert(unit != nullptr);
  //   maybe_add_to_gather_town_hall(unit, receiver);
  // }

  // set defend targets
  for (const auto& units : _enemy_troops) {
    for (const auto& unit : units) {
      assert(unit != nullptr);
      maybe_add_to_defend_targets(unit);
    }
  }

  _result = OK;
}

void Preload::setUnitId2IdxMaps() {
  for (const auto& units : _my_troops) {
    for (const auto& unit : units) {
      UnitId id = unit->GetId();
      // std::cout << "inserting id>>>: " << id  << std::endl;
      // std::cout << "before insert" << std::endl;
      // for (auto it : _id2idx) {
      //   std::cout << it.first << ", " << it.second <<std::endl;
      // }
      // if(_id2idx.find(id) != _id2idx.end()) {
      //   std::cout << _id2idx.at(id) << std::endl;
      //   assert(false);
      // }
      int idx;
      if (unit->GetUnitType() == RESOURCE) {
        idx = _resourceidx2id.size();
        _resourceidx2id.push_back(id);
      } else {
        idx = _myidx2id.size();
        _myidx2id.push_back(id);
      }
      _id2idx[id] = idx;
    }
  }

  for (const auto& units : _enemy_troops) {
    for (const auto& unit : units) {
      UnitId id = unit->GetId();
      assert(_id2idx.find(id) == _id2idx.end());
      _id2idx[id] = _enemyidx2id.size();
      _enemyidx2id.push_back(id);
    }
  }
}

nlohmann::json Preload::log2Json(const GameEnvAspect& aspect,
                                 const CmdReceiver& receiver) const {
  assert(Ready());
  nlohmann::json data;
  data["resource"] = Resource();

  int my_idx = 0;
  int resource_idx = 0;
  data["my_units"] = nlohmann::json::array();
  data["resource_units"] = nlohmann::json::array();
  for (const auto& units : _my_troops) {
    for (const auto& unit : units) {
      std::string key;
      if (unit->GetUnitType() == RESOURCE) {
        key = "resource_units";
        assert(unit->GetId() == _resourceidx2id.at(resource_idx));
        assert(resource_idx == _id2idx.at(unit->GetId()));
        ++resource_idx;
      } else {
        key = "my_units";
        assert(unit->GetId() == _myidx2id.at(my_idx));
        assert(my_idx == _id2idx.at(unit->GetId()));
        ++my_idx;
      }
      data[key].push_back(unit->log2Json(aspect, receiver, _id2idx));
    }
  }

  data["cons_count"] = nlohmann::json::array();
  for (int c : _cnt_under_construction) {
    data["cons_count"].push_back(c);
  }

  int enemy_idx = 0;
  data["enemy_units"] = nlohmann::json::array();
  for (const auto& units : _enemy_troops) {
    for (const auto& unit : units) {
      assert(unit->GetId() == _enemyidx2id.at(enemy_idx));
      assert(enemy_idx == _id2idx.at(unit->GetId()));
      ++enemy_idx;

      data["enemy_units"].push_back(unit->log2Json(aspect, receiver, _id2idx));
    }
  }
  return data;
}

nlohmann::json Preload::partialLog2Json(const GameEnvAspect& aspect,
                                        const CmdReceiver& receiver,
                                        std::list<UnitId> unitIds) const {
  assert(Ready());
  nlohmann::json data;

  int my_idx = 0;
  data["my_units"] = nlohmann::json::array();
  for (const auto& units : _my_troops) {
    for (const auto& unit : units) {
      std::string key;
      if (unit->GetUnitType() == RESOURCE) {
        continue;
      } else {
        UnitId id = unit->GetId();
        auto finder = std::find(unitIds.begin(), unitIds.end(), id);
        if (finder == unitIds.end()) {
          ++my_idx;
          continue;
        }
        unitIds.erase(finder);
        key = "my_units";
        // std::cout << "error?: " <<  _myidx2id.at(my_idx) << std::endl;
        // std::cout << "error?: actual_id: " << unit->GetId() << std::endl;
        assert(unit->GetId() == _myidx2id.at(my_idx));
        assert(my_idx == _id2idx.at(unit->GetId()));
        ++my_idx;
      }
      data[key].push_back(unit->log2Json(aspect, receiver, _id2idx));
    }
  }
  if (unitIds.size()) {
    std::cout << "Warning: unitIds are not exhausted" << std::endl;
  }
  return data;
}
