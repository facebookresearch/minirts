// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once
#include <sstream>

class RTSGameOption {
 public:
  std::string save_replay_prefix;  // "When not empty, save replays to the files
  int save_replay_per_games = 10;  // save replay once per K games
  int save_replay_freq = 0;  // save replay once per K tick (within one game)")
  int seed = 0;              // if seed = 0, then we use time(NULL)")
  int main_loop_quota =
      0;  // ms allowed to spend in main_loop. used to control fps in replay
  int max_tick = 20000;     // max tick per game
  bool no_terrain = false;  // no terrain
  int resource = 500;       // initial resource to start with
  int resource_dist = 4;    // distance to closest resource
  bool fair = true;         // fairness of resource generation
  int num_resources = 3;    // num resource units to start with
  int num_peasants = 3;     // num peasants to start with
  int num_extra_units = 0;  // num of extra units for increased diversity
  int max_num_units_per_player = -1;  // max num of units per player
  int num_games_per_thread = 0;       // max num games per thread
  bool team_play = false;  // flag for teamplay, used in data collection
  std::string lua_files = "./";

  std::string info() const {
    std::stringstream ss;
    ss << std::boolalpha;
    ss << "Save replay prefix: \"" << save_replay_prefix << "\"" << std::endl;
    ss << "Seed: " << seed << std::endl;
    ss << "Main Loop quota: " << main_loop_quota << std::endl;
    ss << "Max ticks: " << max_tick << std::endl;
    ss << "No Terrain: " << no_terrain << std::endl;
    ss << "Resource Dist: " << resource_dist << std::endl;
    ss << "Resource: " << resource << std::endl;
    ss << "Num Resources: " << num_resources << std::endl;
    ss << "Resource Fair: " << fair << std::endl;
    ss << "Num Peasants: " << num_resources << std::endl;
    ss << "Num Extra Unit: " << num_extra_units << std::endl;
    ss << "Max Num Units Per Player: " << max_num_units_per_player << std::endl;
    ss << "Num Games Per Thread: " << num_games_per_thread << std::endl;
    return ss.str();
  }
};
