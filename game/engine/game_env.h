// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include "bullet.h"
#include "cmd_receiver.h"
#include "map.h"
#include "player.h"
#include "unit.h"
#include <nlohmann/json.hpp>
#include <random>

class GameEnv {
 private:
  // Game definitions.
  GameDef _gamedef;

  // Game counter.
  int _game_counter;

  // Next unit_id, initialized to be 0
  UnitId _next_unit_id;

  // Unit hash tables.
  Units _units;

  // Bullet tables.
  Bullets _bullets;

  // The game map.
  std::unique_ptr<RTSMap> _map;

  // Players
  std::vector<Player> _players;

  // Random number generator
  std::mt19937 _rng;

  // Who won the game?
  PlayerId _winner_id;

  // Whether the game is terminated.
  // This happens if the time tick exceeds max_tick, or there is anything wrong.
  bool _terminated;

  // Froze the game in order to issue an instruction
  GameStatus _game_status = ACTIVE_STATUS;

  bool _team_play = false;

  int _max_num_units_per_player;

  std::vector<int> _player_unit_counts;

 public:
  GameEnv(int max_num_units_per_player);

  void Visualize() const;

  // Remove all players.
  void ClearAllPlayers();

  // Reset RTS game to starting position
  void Reset();

  // Add and remove players.
  void AddPlayer(const std::string& name);

  // void RemovePlayer();

  int GetNumOfPlayers() const {
    return _players.size();
  }

  int GetGameCounter() const {
    return _game_counter;
  }

  // Set seed from the random generator.
  void SetSeed(int seed) {
    _rng.seed(seed);
  }

  // Return a random integer from 0 to r - 1
  std::function<uint16_t(int)> GetRandomFunc() {
    return [&](int r) -> uint16_t { return _rng() % r; };
  }

  const RTSMap& GetMap() const {
    return *_map;
  }

  RTSMap& GetMap() {
    return *_map;
  }

  // Generate a RTSMap given number of obstacles.
  bool GenerateMap(int num_obstacles, int init_resource);
  bool GenerateImpassable(int num_obstacles);

  // Generate a maze used by Tower Defense.
  bool GenerateTDMaze();

  const Units& GetUnits() const {
    return _units;
  }

  Units& GetUnits() {
    return _units;
  }

  // Initialize different units for this game.
  void InitGameDef(const std::string& lua_files) {
    _gamedef.Init(lua_files);
  }

  const GameDef& GetGameDef() const {
    return _gamedef;
  }

  GameDef& GetGameDef() {
    return _gamedef;
  }

  // Get a unit from its Id.
  const Unit* GetUnit(UnitId id) const {
    const Unit* target = nullptr;
    if (id != INVALID) {
      auto it = _units.find(id);
      if (it != _units.end())
        target = it->second.get();
    }
    return target;
  }

  Unit* GetUnit(UnitId id) {
    Unit* target = nullptr;
    if (id != INVALID) {
      auto it = _units.find(id);
      if (it != _units.end())
        target = it->second.get();
    }
    return target;
  }

  // Find the closest base.
  UnitId FindFirstTownHall(PlayerId player_id) const;

  UnitId FindClosestTownHall(PlayerId player_id, const PointF& p) const;

  const Unit* FindClosestTownHallP(PlayerId player_id, const PointF& p) const;

  // Find the closest enemy
  UnitId FindClosestEnemy(PlayerId player_id,
                          const PointF& p,
                          float radius) const;

  // Find empty place near a place, used by creating units.
  bool FindEmptyPlaceNearby(const UnitTemplate& unit_def,
                            const PointF& p,
                            int l1_radius,
                            const std::set<PointF>& place_taken,
                            PointF* res_p) const;

  // Find empty place near a place, used by creating buildings.
  bool FindBuildPlaceNearby(const PointF& p,
                            int l1_radius,
                            PointF* res_p) const;

  // Find closest place to a group with a certain distance, used by hit and run.
  bool FindClosestPlaceWithDistance(const Unit& u,
                                    const PointF& p,
                                    int l1_radius,
                                    const std::vector<const Unit*>& units,
                                    PointF* res_p) const;

  const Player& GetPlayer(PlayerId player_id) const {
    assert(player_id >= 0 && player_id < (int)_players.size());
    return _players[player_id];
  }
  Player& GetPlayer(PlayerId player_id) {
    assert(player_id >= 0 && player_id < (int)_players.size());
    return _players[player_id];
  }

  // Add and remove units.
  bool AddUnit(Tick tick,
               UnitType type,
               const PointF& p,
               PlayerId player_id,
               Tick expiration = -1);
  bool RemoveUnit(const UnitId& id);
  void UpdateTemporaryUnits(Tick tick);

  void AddBullet(const Bullet& b) {
    _bullets.push_back(b);
  }

  // Check if one player's town_hall has been destroyed.
  PlayerId CheckTownHall() const;
  // Check if one player has money left to build units or have some units
  PlayerId CheckUnitsAndMoney() const;

  // Getter and setter for winner_id, termination.
  void SetWinnerId(PlayerId winner_id) {
    _winner_id = winner_id;
  }
  PlayerId GetWinnerId() const {
    return _winner_id;
  }
  void SetTermination() {
    _terminated = true;
  }
  bool GetTermination() const {
    return _terminated;
  }

  // Compute bullets cmds.
  void Forward(CmdReceiver* receiver);
  void ComputeFOW();

  // Some debug code.
  int GetPrevSeenCount(PlayerId) const;

  // Fill in metadata to a save_class
  template <typename save_class, typename T>
  void FillHeader(const CmdReceiver& receiver, T* game) const {
    save_class::SetTick(receiver.GetTick(), game);
    save_class::SetWinner(_winner_id, game);
    save_class::SetTermination(_terminated, game);
  }

  // Fill in data to a save_class
  template <typename save_class, typename T>
  void FillIn(PlayerId player_id, const CmdReceiver& receiver, T* game) const;

  void SaveSnapshot(serializer::saver& saver) const;
  void LoadSnapshot(serializer::loader& loader);

  void ChangeGameStatus(GameStatus new_status) {
    if (IsTeamPlay()) {
      _game_status = new_status;
    }
  }

  GameStatus GetGameStatus() const {
    if (IsTeamPlay()) {
      return _game_status;
    }
    return ACTIVE_STATUS;
  }

  void SetTeamPlay(const bool team_play) {
    _team_play = team_play;
  }

  bool IsTeamPlay() const {
    return _team_play;
  }

  bool IsGameActive() const {
    return GetGameStatus() == ACTIVE_STATUS;
  }

  // Compute the hash code.
  uint64_t CurrentHashCode() const;

  std::string PrintDebugInfo() const;

  nlohmann::json LogMap2Json(PlayerId player_id) const;

  ~GameEnv() {
  }
};

class GameEnvAspect {
 public:
  GameEnvAspect(const GameEnv& env, PlayerId player_id, bool respect_fow)
      : _env(env)
      , _player_id(player_id)
      , _respect_fow(respect_fow) {
  }

  bool FilterWithFOW(const Unit& u, bool check_saved_units) const {
    if (!_respect_fow) {
      return true;
    }
    return _env.GetPlayer(_player_id).FilterWithFOW(u, check_saved_units);
  }

  // [TODO] This violates the behavior of Aspect. Will need to change.
  const Units& GetAllUnits() const {
    return _env.GetUnits();
  }

  const Unit* GetUnit(UnitId id) const {
    const Unit* unit = _env.GetUnit(id);
    if (unit == nullptr || FilterWithFOW(*unit, true)) {
      return unit;
    }

    return nullptr;
  }

  const Player& GetPlayer() const {
    return _env.GetPlayer(_player_id);
  }

  const GameDef& GetGameDef() const {
    return _env.GetGameDef();
  }

 private:
  const GameEnv& _env;
  PlayerId _player_id;
  bool _respect_fow;
};

class UnitIterator {
 public:
  enum Type { ALL = 0, BUILDING, MOVING };

  UnitIterator(const GameEnvAspect& aspect, Type type)
      : _aspect(aspect)
      , _type(type) {
    _it = _aspect.GetAllUnits().begin();
    next();
  }

  UnitIterator(const UnitIterator& i) = delete;

  UnitIterator& operator++() {
    ++_it;
    next();
    return *this;
  }

  const Unit& operator*() {
    return *_it->second;
  }

  bool end() const {
    return _it == _aspect.GetAllUnits().end();
  }

 private:
  const GameEnvAspect& _aspect;
  Type _type;
  Units::const_iterator _it;

  void next() {
    while (_it != _aspect.GetAllUnits().end()) {
      const Unit& u = *_it->second;
      if (_aspect.FilterWithFOW(u, true)) {
        if (_type == ALL) {
          break;
        }

        auto unit_type = u.GetUnitType();
        bool is_building = _aspect.GetGameDef().IsUnitTypeBuilding(unit_type);
        if ((is_building && _type == BUILDING) ||
            (!is_building && _type == MOVING)) {
          break;
        }
      }
      ++_it;
    }
  }
};

// Fill in data to a save_class
template <typename save_class, typename T>
void GameEnv::FillIn(PlayerId player_id,
                     const CmdReceiver& receiver,
                     T* game) const {
  bool is_spectator = (player_id == INVALID);

  save_class::SaveGameDef(_gamedef, game);
  save_class::SetPlayerId(player_id, game);
  save_class::SetSpectator(is_spectator, game);
  save_class::SetGameStatus(_game_status, game);

  if (is_spectator) {
    // Show all the maps.
    save_class::Save(*_map, game);
    for (size_t i = 0; i < _players.size(); ++i) {
      save_class::SaveStats(_players[i], game);
    }
    save_class::SavePlayerInstructions(_players[0], game);
  } else {
    // Show only visible maps
    save_class::SavePlayerMap(_players[player_id], game);
    save_class::SaveStats(_players[player_id], game);
    save_class::SavePlayerInstructions(_players[player_id], game);
  }

  GameEnvAspect aspect(*this, player_id, !is_spectator);

  // Building iterator.
  UnitIterator iterator_build(aspect, UnitIterator::BUILDING);
  while (!iterator_build.end()) {
    save_class::Save(*iterator_build, &receiver, game);
    ++iterator_build;
  }

  // Moving iterator.
  UnitIterator iterator_move(aspect, UnitIterator::MOVING);
  while (!iterator_move.end()) {
    save_class::Save(*iterator_move, &receiver, game);
    ++iterator_move;
  }

  for (const auto& bullet : _bullets) {
    save_class::Save(bullet, game);
  }
}
