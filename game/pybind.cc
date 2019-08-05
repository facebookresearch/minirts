// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#include <pybind11/pybind11.h>

#include "engine/game_base.h"

#include "ai/ai.h"
#include "ai/rule_ai.h"
// #include "ai/trainable_rule_ai.h"
// #include "ai/trainable_executor_ai.h"
// #include "ai/mix_executor_ai.h"
#include "ai/cheat_executor_ai.h"

namespace py = pybind11;

PYBIND11_MODULE(minirts, m) {
  py::class_<AIOption>(m, "AIOption")
      .def(py::init<>())
      .def(py::init<const AIOption&>())
      .def("info", &AIOption::info)
      .def_readwrite("type", &AIOption::type)
      .def_readwrite("coach_type", &AIOption::coach_type)
      .def_readwrite("fs", &AIOption::fs)
      .def_readwrite("fow", &AIOption::fow)
      .def_readwrite("resource_scale", &AIOption::resource_scale)
      // .def_readwrite("adapt_resource_scale", &AIOption::adapt_resource_scale)
      // .def_readwrite("win_rate_decay", &AIOption::win_rate_decay)
      // .def_readwrite("min_resource_scale", &AIOption::min_resource_scale)
      // .def_readwrite("max_resource_scale", &AIOption::max_resource_scale)
      .def_readwrite("t_len", &AIOption::t_len)
      .def_readwrite("use_moving_avg", &AIOption::use_moving_avg)
      .def_readwrite("moving_avg_decay", &AIOption::moving_avg_decay)
      .def_readwrite("num_resource_bins", &AIOption::num_resource_bins)
      .def_readwrite("resource_bin_size", &AIOption::resource_bin_size)
      // .def_readwrite("adversarial", &AIOption::adversarial)
      // .def_readwrite("adversarial_decay", &AIOption::adversarial_decay)
      .def_readwrite("log_state", &AIOption::log_state)
      .def_readwrite("verbose", &AIOption::verbose)
      .def_readwrite("max_num_units", &AIOption::max_num_units)
      .def_readwrite("num_prev_cmds", &AIOption::num_prev_cmds)
      .def_readwrite("num_instructions", &AIOption::num_instructions)
      .def_readwrite("max_raw_chars", &AIOption::max_raw_chars);

  py::class_<RTSGameOption>(m, "RTSGameOption")
      .def(py::init<>())
      .def(py::init<const RTSGameOption&>())
      .def("info", &RTSGameOption::info)
      .def_readwrite("save_replay_prefix", &RTSGameOption::save_replay_prefix)
      .def_readwrite("save_replay_per_games", &RTSGameOption::save_replay_per_games)
      .def_readwrite("save_replay_freq", &RTSGameOption::save_replay_freq)
      .def_readwrite("seed", &RTSGameOption::seed)
      .def_readwrite("main_loop_quota", &RTSGameOption::main_loop_quota)
      .def_readwrite("max_tick", &RTSGameOption::max_tick)
      .def_readwrite("no_terrain", &RTSGameOption::no_terrain)
      .def_readwrite("resource", &RTSGameOption::resource)
      .def_readwrite("resource_dist", &RTSGameOption::resource_dist)
      .def_readwrite("fair", &RTSGameOption::fair)
      .def_readwrite("num_resources", &RTSGameOption::num_resources)
      .def_readwrite("num_peasants", &RTSGameOption::num_peasants)
      .def_readwrite("num_extra_units", &RTSGameOption::num_extra_units)
      .def_readwrite("max_num_units_per_player", &RTSGameOption::max_num_units_per_player)
      .def_readwrite("num_games_per_thread", &RTSGameOption::num_games_per_thread)
      .def_readwrite("team_play", &RTSGameOption::team_play)
      .def_readwrite("lua_files", &RTSGameOption::lua_files);

  py::class_<RTSGame, tube::EnvThread, std::shared_ptr<RTSGame>>(m, "RTSGame")
      .def(py::init<const RTSGameOption&>())
      // .def("init_lua", &RTSGame::initLua)
      .def("add_bot", &RTSGame::addBot, py::keep_alive<1, 2>())
      .def("add_default_spectator", &RTSGame::addDefaultSpectator);

  py::class_<AI, std::shared_ptr<AI>>(m, "AI");

  // TOOD: is this intermediate class neceaasry?
  // py::class_<TrainableAI<RuleFeatureExtractor>,
  //            AI,
  //            std::shared_ptr<TrainableAI<RuleFeatureExtractor>>>(m, "TrainableRuleAI_");

  // py::class_<TrainableRuleAI,
  //            AI,
  //            std::shared_ptr<TrainableRuleAI>>(m, "TrainableRuleAI")
  //     .def(py::init<
  //          const AIOption&,              // option
  //          int,                          // threadId
  //          std::shared_ptr<tube::DataChannel>, // trainDc
  //          std::shared_ptr<tube::DataChannel>  // actDc
  //          >());

  // py::class_<TrainableExecutorAI,
  //            AI,
  //            std::shared_ptr<TrainableExecutorAI>>(m, "TrainableExecutorAI")
  //     .def(py::init<
  //          const AIOption&,              // option
  //          int,                          // threadId
  //          std::shared_ptr<tube::DataChannel>, // trainDc
  //          std::shared_ptr<tube::DataChannel>  // actDc
  //          >());

  py::class_<CheatExecutorAI,
             AI,
             std::shared_ptr<CheatExecutorAI>>(m, "CheatExecutorAI")
      .def(py::init<
           const AIOption&,              // option
           int,                          // threadId
           std::shared_ptr<tube::DataChannel>, // trainDc
           std::shared_ptr<tube::DataChannel>  // actDc
           >());

  py::class_<MediumAI, AI, std::shared_ptr<MediumAI>>(m, "MediumAI")
      .def(py::init<
           const AIOption&,
           int,
           std::shared_ptr<tube::DataChannel>,
           UnitType,
           bool>());

  py::enum_<UnitType>(m, "UnitType", py::arithmetic())
      .value("INVALID_UNITTYPE", INVALID_UNITTYPE)
      .value("RESOURCE", RESOURCE)
      .value("PEASANT", PEASANT)
      .value("SPEARMAN", SPEARMAN)
      .value("SWORDMAN", SWORDMAN)
      .value("CAVALRY", CAVALRY)
      .value("DRAGON", DRAGON)
      .value("ARCHER", ARCHER)
      .value("CATAPULT", CATAPULT)
      .value("BARRACK", BARRACK)
      .value("BLACKSMITH", BLACKSMITH)
      .value("STABLE", STABLE)
      .value("WORKSHOP", WORKSHOP)
      .value("AVIARY", AVIARY)
      .value("ARCHERY", ARCHERY)
      .value("GUARD_TOWER", GUARD_TOWER)
      .value("TOWN_HALL", TOWN_HALL)
      .value("NUM_MINIRTS_UNITTYPE", NUM_MINIRTS_UNITTYPE)
      .export_values();
};
