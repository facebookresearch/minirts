// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include <sstream>

// TODO: split this option
class AIOption {
 public:
  std::string type = ""; //type of ai");
  std::string coach_type = ""; //coach type, optional");

  int fs = 1; //frame skip");
  bool fow = true; //whether ai respects fog-of-war");

  float resource_scale = 1.0; //adjust resource scale, can be used as handicap");
  // bool adapt_resource_scale = false; //whether to use adaptive resource scale");
  // float win_rate_decay = 1.0; //for running average of win rate");
  // float min_resource_scale = 0.5; //min resource scale");
  // float max_resource_scale = 1.2; //max resource scale");

  int t_len = -1; //trajectory length for actor critic");

  bool use_moving_avg = false; //whether use moving average feature");
  float moving_avg_decay = 0.0; //decay of moving average");
  int num_resource_bins = -1; //, "num resource bins");
  int resource_bin_size = -1; //, "resource bin size");

  // float adversarial = 0.0; //whether use adversarial opponent selection");
  // float adversarial_decay = 0.0; //decay of per type win rate");

  bool log_state = false; //store state for behavior cloning");

  // related to trainable executor ai
  // TODO: this is horrible design, should seprate to different options
  int max_num_units = -1;
  int num_prev_cmds = -1;
  int num_instructions = -1;
  int max_raw_chars = -1;
  bool verbose = false;

  std::string info() const {
    std::stringstream ss;
    ss << std::boolalpha
       << "[type: " << type << "]\n"
       << "[coach_type: " << coach_type << "]\n"
       << "[fs: " << fs << "]\n"
       << "[fow: " << fow << "]\n"
       << "[resource_scale: " << resource_scale << "]\n"
       // << "[adapt_resource_scale: " << adapt_resource_scale << "]\n"
       // << "[win_rate_decay: " << win_rate_decay << "]\n"
       // << "[min_resource_scale: " << min_resource_scale << "]\n"
       // << "[max_resource_scale: " << max_resource_scale << "]\n"
       << "[t_len: " << t_len << "]\n"
       << "[use_moving_avg: " << use_moving_avg << "]\n"
       << "[moving_avg_decay: " << moving_avg_decay << "]\n"
       << "[num_resource_bins: " << num_resource_bins << "]\n"
       << "[resource_bin_size: " << resource_bin_size << "]\n"
       // << "[adversarial: " << adversarial << "]\n"
       // << "[adversarial_decay: " << adversarial_decay << "]\n"
       << "[log_state: " << log_state << "]\n"
       << "[max_num_units: " << max_num_units << "]\n"
       << "[num_prev_cmds: " << num_prev_cmds << "]\n"
       << "[num_instructions: " << num_instructions << "]\n"
       << "[max_raw_chars: " << max_raw_chars << "]\n"
       << "[verbose: " << verbose << "]\n";

    return ss.str();
  }

};
