// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include <nlohmann/json.hpp>
#include "engine/common.h"
#include "engine/gamedef.h"

using json = nlohmann::json;

class RTSMap;
class Player;
class Unit;
class Bullet;
class CmdReceiver;

class save2json {
 public:
  static void SetTick(Tick tick, json* game);
  static void SetWinner(PlayerId id, json* game);
  static void SetTermination(bool t, json* game);
  static void SetGameCounter(int game_counter, json* game);
  static void SetPlayerId(PlayerId id, json* game);
  static void SetSpectator(bool is_spectator, json* game);
  static void SetGameStatus(GameStatus game_status, json* game);

  static void Save(const RTSMap& m, json* game);
  static void SaveStats(const Player& player, json* game);
  static void SavePlayerMap(const Player& player, json* game);
  static void SavePlayerInstructions(const Player& player, json* game);
  static void SaveGameDef(const GameDef& gamedef, json* game);
  // static void Save(const AI &bot, json *game);
  static void Save(const Unit& unit, const CmdReceiver* receiver, json* game);
  static void Save(const Bullet& bullet, json* game);
};
