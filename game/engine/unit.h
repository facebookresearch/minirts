// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include <assert.h>
#include <initializer_list>
#include <nlohmann/json.hpp>

#include "cmd.h"
#include "gamedef.h"
#include "map.h"

class GameEnvAspect;

class Unit {
 protected:
  // Basic Attributes.
  UnitId _id;
  UnitType _type;
  PointF _p, _last_p;
  Tick _built_since;
  UnitProperty _property;
  Tick _expiration;

 public:
  Unit()
      : Unit(INVALID, INVALID, PEASANT, PointF(), UnitProperty()) {
  }

  Unit(Tick tick,
       UnitId id,
       UnitType t,
       const PointF& p,
       const UnitProperty& property,
       Tick expiration = -1)
      : _id(id)
      , _type(t)
      , _p(p)
      , _last_p(p)
      , _built_since(tick)
      , _property(property)
      , _expiration(expiration) {
    // set progress to 0 for units under construction
    if (IsTemporary()) {
      _property._hp = 0;
    }
  }

  nlohmann::json log2Json(const GameEnvAspect& aspect,
                          const CmdReceiver& receiver,
                          const std::map<UnitId, int>& id2idx) const;

  bool operator<(const Unit& other) const {
    if (_type < other._type) {
      return true;
    }
    if (_id < other._id) {
      return true;
    }
    if (_built_since < other._built_since) {
      return true;
    }
    return false;
  }

  bool operator==(const Unit& other) const {
    return _type == other._type && _id == other._id;
  }

  UnitId GetId() const {
    return _id;
  }

  PlayerId GetPlayerId() const;

  const PointF& GetPointF() const {
    return _p;
  }

  PointF GetCorrectedPointF() const {
    PointF p(std::max((float)0, _p.x), std::max((float)0, _p.y));
    return p;
  }

  const PointF& GetLastPointF() const {
    return _last_p;
  }

  void SetPointF(const PointF& p) {
    _last_p = _p;
    _p = p;
  }

  Tick GetBuiltSince() const {
    return _built_since;
  }

  UnitType GetUnitType() const {
    return _type;
  }

  float GetNormalizedHp() const {
    return _property._hp / _property._max_hp;
  }

  bool IsResource() const {
    return _type == RESOURCE;
  }

  bool IsUnit() const {
    return _type == PEASANT || _type == SPEARMAN || _type == SWORDMAN ||
           _type == CAVALRY || _type == DRAGON || _type == ARCHER ||
           _type == CATAPULT;
  }

  bool IsBuilding() const {
    return GameDef::IsUnitTypeBuilding(_type);
  }

  const UnitProperty& GetProperty() const {
    return _property;
  }

  UnitProperty& GetProperty() {
    return _property;
  }

  bool IsTemporary() const {
    return _expiration != -1;
  }

  void ChangeExpiration(int new_expiration) {
    _expiration = new_expiration;
  }

  bool UpdateBuildProgress(Tick tick) {
    assert(_expiration != -1);

    // update build progress
    const double progress = (tick - _built_since) / (double)_expiration;
    _property._hp = std::min(
        _property._max_hp, static_cast<int>(progress * _property._max_hp));

    // return whether the building has been finished
    return tick - _built_since >= _expiration;
  }

  // Visualization. Output a vector of string as the visual command.
  std::string Draw(Tick tick) const;

  // Print info in the screen.
  std::string PrintInfo() const;

  SERIALIZER(Unit, _id, _type, _p, _last_p, _built_since, _property);
  HASH(Unit, _property, _id, _type, _p, _last_p, _built_since);
};

STD_HASH(Unit);

using Units = std::map<UnitId, std::unique_ptr<Unit>>;
