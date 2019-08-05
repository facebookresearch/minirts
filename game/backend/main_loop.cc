// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#include "ai/ai.h"
#include "ai/ai_factory.h"
#include "ai/comm_ai.h"
#include "ai/replay_loader.h"

#include "engine/cmd_util.h"
#include "engine/common.h"
#include "engine/game_base.h"
#include "engine/game_state_ext.h"
#include "engine/unit.h"
#include "engine/utils.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <thread>
#include <unordered_map>
#include <vector>

using Parser = CmdLineUtils::CmdLineParser;

bool add_players(const std::string& args,
                 RTSGame* game,
                 int player_port,
                 int coach_port,
                 int spectator_port,
                 int seed) {
  std::vector<TCPAI*> tcp_agents;
  std::cout << ">>> args: " << args << std::endl;

  for (const auto& player : split(args, ';')) {
    std::cout << "Dealing with player = " << player << std::endl;
    std::vector<std::string> params = split(player, '=');
    if (player.find("tcp_team") == 0) {
      // create player ai
      auto player_ai = std::make_unique<TCPPlayerAI>(player_port);
      tcp_agents.push_back(player_ai.get());
      const int player_id = game->addBot(std::move(player_ai));
      // create coach ai
      auto coach_ai = std::make_unique<TCPCoachAI>(coach_port, player_id);
      tcp_agents.push_back(coach_ai.get());
      game->addBot(std::move(coach_ai));
    } else if (player.find("tcpai_team") == 0) {
      // create player ai
      auto player_ai = std::make_unique<TCPPlayerAI>(player_port);
      tcp_agents.push_back(player_ai.get());
      game->addBot(std::move(player_ai));
      // create coach ai
      auto coach_ai =
          std::shared_ptr<AI>(AIFactory::CreateAI("coach", seed, Params()));
      game->addBot(std::move(coach_ai));
    } else if (player.find("tcp") == 0) {
      auto ai = std::make_unique<TCPAI>(player_port);
      ++player_port;
      tcp_agents.push_back(ai.get());
      game->addBot(std::move(ai));
    } else if (player.find("spectator") == 0) {
      int tick_start = (params.size() >= 2 ? 0 : std::stoi(params[1]));
      std::string replay_name = (params.size() >= 3 ? params[2] : "");
      auto spectator = std::make_unique<TCPSpectator>(
          replay_name, tick_start, spectator_port);
      game->addSpectator(std::move(spectator));
    } else if (player.find("replayer") == 0) {
      assert(params.size() == 2);
      std::string replay_name = params[1];
      auto replayer = std::make_unique<Replayer>(replay_name);
      game->addSpectator(std::move(replayer));
    } else if (player.find("dummy") == 0) {
      game->addBot(std::make_unique<AI>(AIOption(), 0));
    } else {
      params = split(player, ',');
      std::string name = params[0];
      Params params_;
      for (size_t i = 1; i < params.size(); ++i) {
        std::vector<std::string> key_val = split(params[i], '=');
        params_[key_val[0]] = key_val[1];
      }

      std::shared_ptr<AI> ai(AIFactory::CreateAI(name, seed, params_));
      if (ai != nullptr) {
        game->addBot(std::move(ai));
      } else {
        std::cout << "Unknown player! " << player << std::endl;
        return false;
      }
    }
  }

  std::cout << "Connecting to TCP players" << std::endl;
  bool all_ready = false;
  for (int attempt = 0; attempt < 250; ++attempt) {
    all_ready = true;
    for (auto& agent : tcp_agents) {
      if (!agent->isReady()) {
        all_ready = false;
        break;
      }
    }
    if (all_ready) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  if (!all_ready) {
    for (auto& agent : tcp_agents) {
      if (!agent->isReady()) {
        std::cout << "client has not connected to port " << agent->port
                  << std::endl;
      }
    }
    throw;
  }
  std::cout << "Done with adding players" << std::endl;
  return true;
}

RTSGameOption GetOption(const Parser& parser) {
  RTSGameOption option;
  int vis_after = parser.GetItem<int>("vis_after", -1);

  option.num_games_per_thread =
      parser.GetItem<bool>("loop", false) ? 100000 : 1;
  option.lua_files = parser.GetItem<std::string>("lua_files", ".");

  option.main_loop_quota = vis_after >= 0 ? 20 : 0;
  option.save_replay_prefix =
      parser.GetItem<std::string>("save_replay_prefix", "");
  option.save_replay_freq = parser.GetItem<int>("save_replay_freq", 100);
  option.max_tick = parser.GetItem<int>("max_tick", 100000);
  option.seed = parser.GetItem<int>("seed");
  option.no_terrain = parser.GetItem<bool>("no_terrain", false);
  option.resource = parser.GetItem<int>("resource", 300);
  option.resource_dist = parser.GetItem<int>("resource_dist", 4);
  option.num_resources = parser.GetItem<int>("num_resources", 3);
  option.num_peasants = parser.GetItem<int>("num_peasants", 3);
  option.fair = parser.GetItem<bool>("fair", true);
  option.num_extra_units = parser.GetItem<int>("num_extra_units", 0);
  option.max_num_units_per_player =
      parser.GetItem<int>("max_num_units_per_player", 0);

  return option;
}

RTSGameOption ai_vs_team(const Parser& parser, std::string* players) {
  RTSGameOption option = GetOption(parser);
  *players = "tcp_team;" + parser.GetItem<std::string>("players");
  option.main_loop_quota = 40;
  option.team_play = true;
  return option;
}

RTSGameOption ai_vs_teamai(const Parser& parser, std::string* players) {
  RTSGameOption option = GetOption(parser);
  *players = "tcpai_team;" + parser.GetItem<std::string>("players");
  option.main_loop_quota = 40;
  option.team_play = true;
  return option;
}

RTSGameOption ai_vs_human(const Parser& parser, std::string* players) {
  RTSGameOption option = GetOption(parser);
  *players = "tcp;" + parser.GetItem<std::string>("players");
  option.main_loop_quota = 40;

  return option;
}

RTSGameOption human_vs_human(const Parser& parser, std::string* players) {
  RTSGameOption option = GetOption(parser);
  *players = "tcp;tcp";
  option.main_loop_quota = 40;

  return option;
}

RTSGameOption ai_vs_ai(const Parser& parser, std::string* players) {
  RTSGameOption option = GetOption(parser);
  *players = parser.GetItem<std::string>("players");
  int vis_after = parser.GetItem<int>("vis_after");
  if (vis_after >= 0) {
    *players += ";spectator=" + std::to_string(vis_after);
  }
  return option;
}

RTSGameOption replay_(const Parser& parser,
                      std::string* players,
                      bool teamplay) {
  RTSGameOption option = GetOption(parser);
  auto arg_players = split(parser.GetItem<std::string>("players"), ';');
  assert(arg_players.size() == 2);
  std::string replay_file = parser.GetItem<std::string>("load_replay");
  std::string replay = ",replay=" + replay_file;
  *players = arg_players[0] + replay;
  if (teamplay) {
    (*players) += ";dummy;";
  } else {
    (*players) += ";";
  }
  *players += (arg_players[1] + replay);

  int vis_after = parser.GetItem<int>("vis_after");
  if (vis_after >= 0) {
    *players += ";spectator=" + std::to_string(vis_after) + "=" + replay_file;
  } else {
    *players += ";replayer=" + replay_file;
  }
  return option;
}

RTSGameOption replay(const Parser& parser, std::string* players) {
  return replay_(parser, players, false);
}

RTSGameOption teamreplay(const Parser& parser, std::string* players) {
  return replay_(parser, players, true);
}

void test() {
  RTSMap m;
  std::vector<Player> players;
  for (int i = 0; i < 2; ++i) {
    players.emplace_back(m, std::to_string(i), i);
  }

  serializer::saver saver(false);
  saver << players;
  if (!saver.write_to_file("tmp.txt")) {
    std::cout << "Write file error!" << std::endl;
  }

  serializer::loader loader(false);
  if (!loader.read_from_file("tmp.txt")) {
    std::cout << "Read file error!" << std::endl;
  }
  loader >> players;

  for (const auto& p : players) {
    std::cout << p.PrintInfo() << std::endl;
  }
}

int main(int argc, char* argv[]) {
  const std::map<std::string,
                 std::function<RTSGameOption(const Parser&, std::string*)>>
      func_mapping = {
          {"selfplay", ai_vs_ai},
          {"replay", replay},
          {"teamreplay", teamreplay},
          {"humanplay", ai_vs_human},
          {"teamplay", ai_vs_team},
          {"teamaiplay", ai_vs_teamai},
          {"humanhuman", human_vs_human},
      };

  CmdLineUtils::CmdLineParser parser(
      "playstyle                                \
           --save_replay_prefix                 \
           --load_replay                        \
           --vis_after[-1]                      \
           --seed[0]                            \
           --max_tick[30000]                    \
           --games[16]                          \
           --players                            \
           --threads[64]                        \
           --load_binary_string                 \
           --player_port                        \
           --coach_port                         \
           --spectator_port                     \
           --lua_files                          \
           --no_terrain                         \
           --resource                           \
           --resource_dist                      \
           --num_resources                      \
           --fair                               \
           --num_peasants                       \
           --num_extra_units                    \
           --loop                               \
           --handicap_level[0]                  \
           --max_num_units_per_player[-1]");

  if (!parser.Parse(argc, argv)) {
    std::cout << parser.PrintHelper() << std::endl;
    return 0;
  }

  std::cout << "Cmd Option: " << std::endl;
  std::cout << parser.PrintParsed() << std::endl;
  auto playstyle = parser.GetItem<std::string>("playstyle");
  auto it = func_mapping.find(playstyle);
  if (it == func_mapping.end()) {
    std::cout << "Unknown command " << playstyle
              << "! Available commands are: " << std::endl;
    for (auto it = func_mapping.begin(); it != func_mapping.end(); ++it) {
      std::cout << it->first << std::endl;
    }
    return 0;
  }

  const std::string lua_files = parser.GetItem<std::string>("lua_files", ".");
  GameDef::GlobalInit(lua_files);

  RTSGameOption option;
  std::string players;
  try {
    if (it->second != nullptr)
      option = it->second(parser, &players);
  } catch (const std::exception& e) {
    std::cout << e.what() << std::endl;
    return 1;
  }

  int player_port = parser.GetItem<int>("player_port", 8000);
  int coach_port = parser.GetItem<int>("coach_port", 8001);
  int spectator_port = parser.GetItem<int>("spectator_port", 8002);

  auto time_start = std::chrono::system_clock::now();

  RTSGame game(option);

  int seed = option.seed;
  if (seed == 0 && playstyle.find("replay") == std::string::npos) {
    seed = get_time_microseconds_mod_by();
  }
  std::cout << "seed:" << seed << std::endl;

  std::cout << "Players: " << players << std::endl;
  add_players(players, &game, player_port, coach_port, spectator_port, seed);
  std::cout << "Finish adding players" << std::endl;

  for (int game_id = 0; game_id < option.num_games_per_thread; ++game_id) {
    std::cout << "Game: " << game_id << std::endl;
    std::chrono::duration<double> duration =
        std::chrono::system_clock::now() - time_start;
    std::cout << "Total time spent = " << duration.count() << "s" << std::endl;
    try {
      game.mainLoop();
    } catch (WSServerException& e) {
      std::cout << e.what() << " port " << e.get_port() << std::endl;
    }
  }

  return 0;
}
