// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include <list>
#include <nlohmann/json.hpp>

#include "cmd.h"
#include "cmd_interface.h"
#include "cmd_receiver.h"
#include "cmd_specific.gen.h"
#include "game_env.h"
#include "unit.h"

// helper class for managing units centered by town_halls
class TownHall {
 public:
  TownHall(const Unit* town_hall, int num_unit_type)
      : _town_hall(town_hall)
      , _closest_resource(nullptr)
      , _dsqr_to_resource(0.0)
      , _num_peasant(0) {
    assert(town_hall != nullptr);
    assert(town_hall->GetUnitType() == TOWN_HALL);
    _nearby_troops.resize(num_unit_type);
  }

  const Unit* GetTownHall() const {
    return _town_hall;
  }

  UnitId GetId() const {
    return _town_hall->GetId();
  }

  UnitId GetResourceId() const {
    if (_closest_resource == nullptr) {
      return INVALID;
    } else {
      return _closest_resource->GetId();
    }
  }

  PlayerId GetPlayerId() const {
    return _town_hall->GetPlayerId();
  }

  const PointF& GetPointF() const {
    return _town_hall->GetPointF();
  }

  void AddUnit(const Unit* u) {
    UnitType unit_type = u->GetUnitType();
    assert(unit_type <= (int)_nearby_troops.size());
    _nearby_troops[unit_type].push_back(u);
    // _all_nearby_troops.push_back(u);
    if (unit_type == RESOURCE) {
      float dsqr = PointF::L2Sqr(u->GetPointF(), _town_hall->GetPointF());
      if (_closest_resource == nullptr || dsqr < _dsqr_to_resource) {
        _dsqr_to_resource = dsqr;
        _closest_resource = u;
      }
    }
  }

  int UnitCount(UnitType unit_type) {
    assert(unit_type <= (int)_nearby_troops.size());
    return _nearby_troops[unit_type].size();
  }

  int AddPeasant() {
    _num_peasant += 1;  //++_num_peasant;
    return _num_peasant;
  }

  int GetNumPeasant() const {
    return _num_peasant;
  }

 private:
  const Unit* _town_hall;
  const Unit* _closest_resource;
  float _dsqr_to_resource;
  int _num_peasant;  // num of peasant uses this town_hall as gather dest
  std::vector<std::vector<const Unit*>> _nearby_troops;
};

// Class to preload information from game environment for future use.
class Preload {
 public:
  enum Result { NOT_READY = -1, OK = 0, NO_TOWN_HALL };

  Preload() {
  }

  void GatherInfo(const GameEnv& env,
                  int player_id,
                  const CmdReceiver& receiver,
                  const std::list<UnitType>& build_queue,
                  bool respect_fow);

  bool Ready() const {
    return _result != NOT_READY;
  }

  bool Ok() const {
    return _result == OK;
  }

  int NumUnit(UnitType ut, bool count_constructing) const {
    int count = _my_troops.at(ut).size();
    if (count_constructing) {
      count += _cnt_under_construction.at(ut);
    }
    return count;
  }

  bool BuildIfAffordable(UnitType ut) {
    if (_resource >= _prices[ut]) {
      _resource -= _prices[ut];
      return true;
    } else {
      return false;
    }
  }

  bool Affordable(UnitType ut) const {
    return _resource >= _prices[ut];
  }

  // budget is used for planning what to build
  bool WithinBudget(UnitType ut) const {
    // (void)ut;
    // return true;
    return _budget >= _prices[ut];
  }

  void DeductBudget(UnitType ut) {
    // (void)ut;
    // return;
    _budget -= _prices[ut];
  }

  int NumTownHallWithResource(bool count_constructing) const {
    // assume new town_hall under construction has resource
    int count = 0;
    if (count_constructing) {
      count = _cnt_under_construction.at(TOWN_HALL);
    }
    for (int i = 0; i < (int)_town_halls.size(); ++i) {
      if (_town_halls[i].GetResourceId() != INVALID) {
        ++count;
      }
    }
    return count;
  }

  TownHall* GetDestinationForGather() {
    int min_num_peasant = 0;
    int dest_idx = -1;
    for (int i = 0; i < (int)_town_halls.size(); ++i) {
      auto town_hall = _town_halls[i];
      if (town_hall.GetResourceId() == INVALID) {
        continue;
      }
      if (dest_idx == -1 || town_hall.GetNumPeasant() < min_num_peasant) {
        dest_idx = i;
        min_num_peasant = town_hall.GetNumPeasant();
      }
    }
    if (dest_idx == -1) {
      return nullptr;
    }
    return &_town_halls.at(dest_idx);
  }

  const TownHall* FindClosestTownHall(const Unit& unit) const {
    float min_dsqr = 0;
    int min_idx = -1;
    for (int i = 0; i < (int)_town_halls.size(); ++i) {
      float dsqr = PointF::L2Sqr(unit.GetPointF(), _town_halls[i].GetPointF());
      if (min_idx == -1 || dsqr < min_dsqr) {
        min_idx = i;
        min_dsqr = dsqr;
      }
    }
    if (min_idx == -1) {
      return nullptr;
    }
    return &_town_halls.at(min_idx);
  }

  int Price(UnitType ut) const {
    return _prices[ut];
  }

  int Resource() const {
    return _resource;
  }

  const std::vector<TownHall>& MyTownHalls() const {
    return _town_halls;
  }

  const std::vector<std::vector<const Unit*>>& MyTroops() const {
    return _my_troops;
  }

  const std::vector<std::vector<const Unit*>>& EnemyTroops() const {
    assert(_enemy_troops[(int)RESOURCE].empty());
    return _enemy_troops;
  }

  const std::vector<const Unit*>& EnemyAllUnits() const {
    return _enemy_all_units;
  }

  // enemy base targets, excluding guard tower
  const std::vector<const Unit*>& EnemyBaseTargets() const {
    return _enemy_base_targets;
  }

  // enemy army at our base, including guard tower
  const std::vector<const Unit*>& EnemyDefendTargets() const {
    return _enemy_defend_targets;
  }

  // enemy moving + guard tower targets
  const std::vector<const Unit*>& EnemyInRangeTargets() const {
    return _enemy_in_range_targets;
  }

  const std::vector<const Unit*>& MyArmy() const {
    return _my_army;
  }

  const std::vector<int>& CntUnderConstruction() const {
    return _cnt_under_construction;
  }

  nlohmann::json log2Json(const GameEnvAspect& aspect,
                          const CmdReceiver& receiver) const;

  nlohmann::json partialLog2Json(const GameEnvAspect& aspect,
                                 const CmdReceiver& receiver,
                                 std::list<UnitId> unitIds) const;

  void setUnitId2IdxMaps();

  const std::map<UnitId, int>& getUnitId2Idx() const {
    return _id2idx;
  }

  const std::vector<int>& getMyidx2id() const {
    return _myidx2id;
  }

  const std::vector<int>& getEnemyidx2id() const {
    return _enemyidx2id;
  }

  const std::vector<int>& getResourceidx2id() const {
    return _resourceidx2id;
  }

 private:
  std::vector<std::vector<const Unit*>> _my_troops;
  std::vector<std::vector<const Unit*>> _enemy_troops;
  std::vector<const Unit*> _enemy_all_units;
  std::vector<const Unit*> _enemy_base_targets;
  std::vector<const Unit*> _enemy_defend_targets;
  std::vector<const Unit*> _enemy_in_range_targets;
  std::vector<const Unit*> _my_army;
  std::vector<int> _cnt_under_construction;
  std::vector<TownHall> _town_halls;
  std::vector<int> _prices;
  int _resource;
  int _budget;
  int _num_unit_type;
  Result _result = NOT_READY;

  std::map<UnitId, int> _id2idx;
  std::vector<UnitId> _myidx2id;
  std::vector<UnitId> _resourceidx2id;
  std::vector<UnitId> _enemyidx2id;

  void collect_stats(const GameEnv& env,
                     int player_id,
                     const CmdReceiver& receiver,
                     bool respect_fow);

  UnitId add_to_closest_town_hall(const Unit* unit);

  bool maybe_add_to_defend_targets(const Unit* unit);

  bool maybe_add_to_gather_town_hall(const Unit* unit,
                                     const CmdReceiver& receiver);
};
