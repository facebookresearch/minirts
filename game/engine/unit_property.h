// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include "nlohmann/json.hpp"
#include <sstream>
#include <string>

#include "common.h"

// [TODO]: Make it more modular.
// TODO: naming
struct UnitProperty {
  int _hp;
  int _max_hp;

  int _att;
  int _def;
  int _att_r;
  float _speed;

  // Visualization range.
  int _vis_r;

  int _changed_hp;
  UnitId _damage_from;

  // Attributes
  UnitAttr _attr;

  // All CDs.
  std::vector<Cooldown> _cds;

  bool IsDead() const {
    return _hp <= 0;
  }

  Cooldown& CD(CDType t) {
    return _cds[t];
  }

  const Cooldown& CD(CDType t) const {
    return _cds[t];
  }

  UnitId GetLastDamageFrom() const {
    return _damage_from;
  }
  void SetCooldown(int t, Cooldown cd) {
    _cds[static_cast<CDType>(t)] = cd;
  }

  // setters for lua
  void SetSpeed(double speed) {
    _speed = static_cast<float>(speed);
  }

  void SetAttr(int attr) {
    _attr = static_cast<UnitAttr>(attr);
  }

  std::string Draw(Tick tick) const {
    std::stringstream ss;
    ss << "Tick: " << tick << " ";
    ss << "H: " << _hp << "/" << _max_hp << " ";
    for (int i = 0; i < NUM_COOLDOWN; ++i) {
      CDType t = (CDType)i;
      int cd_val = CD(t)._cd;
      int diff = std::min(tick - CD(t)._last, cd_val);
      ss << t << " [last=" << CD(t)._last << "][diff=" << diff
         << "][cd=" << cd_val << "]; ";
    }
    return ss.str();
  }

  std::string PrintInfo() const {
    return std::to_string(_hp) + "/" + std::to_string(_max_hp);
  }

  nlohmann::json log2Json() const {
    nlohmann::json data;
    data["hp"] = _hp;
    data["max_hp"] = _max_hp;
    data["att"] = _att;
    data["def"] = _def;
    data["att_r"] = _att_r;
    data["speed"] = _speed;
    data["vis_r"] = _vis_r;
    data["changed_hp"] = _changed_hp;
    data["damage_from"] = _damage_from;
    data["cds"] = nlohmann::json::array();
    for (auto cd : _cds) {
      nlohmann::json cd_data;
      cd_data["last"] = cd._last;
      cd_data["cd"] = cd._cd;
      data["cds"].push_back(cd_data);
    }
    return data;
  }

  UnitProperty()
      : _hp(0)
      , _max_hp(0)
      , _att(0)
      , _def(0)
      , _att_r(0)
      , _speed(0.0)
      , _vis_r(0)
      , _changed_hp(0)
      , _damage_from(INVALID)
      , _attr(ATTR_NORMAL)
      , _cds(NUM_COOLDOWN) {
  }

  SERIALIZER(UnitProperty,
             _hp,
             _max_hp,
             _att,
             _def,
             _att_r,
             _speed,
             _vis_r,
             _changed_hp,
             _damage_from,
             _attr,
             _cds);

  HASH(UnitProperty,
       _hp,
       _max_hp,
       _att,
       _def,
       _att_r,
       _speed,
       _vis_r,
       _changed_hp,
       _damage_from,
       _attr,
       _cds);
};

STD_HASH(UnitProperty);
