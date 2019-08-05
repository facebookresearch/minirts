// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#include "unit.h"
#include "cmd_target.h"
#include "utils.h"
#include <sstream>

// -----------------------  Unit definition ----------------------
PlayerId Unit::GetPlayerId() const {
  return ExtractPlayerId(_id);
}

std::string Unit::PrintInfo() const {
  // Print the information for the unit.
  std::stringstream ss;
  ss << "U[" << ExtractPlayerId(_id) << ":" << _id << "], @(" << _p.x << ", "
     << _p.y << "), ";
  ss << _type << ", H " << _property._hp << "/" << _property._max_hp << ", B "
     << _built_since << " A" << _property._att << " D" << _property._def
     << " | ";
  for (int j = 0; j < NUM_COOLDOWN; j++) {
    const auto& cd = _property.CD((CDType)j);
    ss << (CDType)j << ": " << cd._cd << "/" << cd._last << "  ";
  }
  return ss.str();
}

std::string Unit::Draw(Tick tick) const {
  // Draw the unit.
  auto unit = make_string("c", ExtractPlayerId(_id), _last_p, _p, _type) + " " +
              _property.Draw(tick);
  return unit;
}

nlohmann::json Unit::log2Json(const GameEnvAspect& aspect,
                              const CmdReceiver& receiver,
                              const std::map<UnitId, int>& id2idx) const {
  nlohmann::json data;
  data["idx"] = id2idx.at(_id);
  data["unit_id"] = _id;
  data["unit_type"] = int(_type);
  data["x"] = _p.x;
  data["y"] = _p.y;
  data["hp"] = float(_property._hp) / float(_property._max_hp);

  CmdTarget current_cmd = CreateCmdTargetForUnit(aspect, receiver, *this);
  if (_type != RESOURCE) {
    data["current_cmd"] = current_cmd.log2Json(id2idx);
  }
  // data["property"] = _property.log2Json();
  return data;
}
