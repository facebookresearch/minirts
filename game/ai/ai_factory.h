// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>

#include "engine/cmd_receiver.h"
// #include "engine/game_state.h"

// #include "elf/ai/ai.h"
#include "ai/ai.h"

using Params = std::map<std::string, std::string>;

// template <typename AI>
class AIFactory {
 public:
  using RegFunc = std::function<AI*(int seed, const Params& params)>;

  // Factory method given specification.
  static AI* CreateAI(const std::string& name, int seed, const Params& params) {
    // std::lock_guard<std::mutex> lock(_mutex);
    auto it = _factories.find(name);
    if (it == _factories.end()) {
      return nullptr;
    }
    return it->second(seed, params);
  }

  static void RegisterAI(const std::string& name, RegFunc reg_func) {
    // std::lock_guard<std::mutex> lock(_mutex);
    _factories.insert(make_pair(name, reg_func));
  }

 private:
  static std::map<std::string, RegFunc> _factories;
  // inline static std::mutex _mutex;
};

// template <typename AI>
// std::map<std::string, AIFactory<AI>::RegFunc> AIFactory<AI>::_factories;
// std::map<std::string, AIFactory::RegFunc> AIFactory::_factories;


template <typename T>
inline bool _WarnIfNoExist(
    const Params& params, const std::string& field, T defaultVal) {
  if (params.find(field) == params.end()) {
    std::cout << "Warning>>>>>: " << field
              << " is not set. Fall back to default value: "
              << defaultVal << std::endl;
    return true;
  }
  return false;
}

inline int GetIntParam(
    const Params& params, const std::string& field, int defaultVal) {
  if (_WarnIfNoExist<int>(params, field, defaultVal)) {
    return defaultVal;
  }

  std::string s = params.at(field);
  return std::stoi(s);
}

inline float GetFloatParam(
    const Params& params, const std::string& field, float defaultVal) {
  if (_WarnIfNoExist<float>(params, field, defaultVal)) {
    return defaultVal;
  }

  std::string s = params.at(field);
  return std::stof(s);
}

inline bool GetBoolParam(
    const Params& params, const std::string& field, bool defaultVal) {
  if (_WarnIfNoExist<bool>(params, field, defaultVal)) {
    return defaultVal;
  }

  std::string s = params.at(field);
  return std::stof(s) > 0;
}

inline std::string GetStringParam(
    const Params& params, const std::string& field, std::string defaultVal) {
  if (_WarnIfNoExist<std::string>(params, field, defaultVal)) {
    return defaultVal;
  }

  return params.at(field);
}

inline UnitType GetUnitTypeParam(
    const Params& params, const std::string& field, UnitType defaultVal) {
  if (_WarnIfNoExist<UnitType>(params, field, defaultVal)) {
    return defaultVal;
  }
  std::string s = params.at(field);
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
      return std::toupper(c);
    });
  return _string2UnitType(s);
}
