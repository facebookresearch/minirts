// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#include "game_env.h"
#include "cmd.h"
#include "cmd_specific.gen.h"
#include "utils.h"

GameEnv::GameEnv(int max_num_units_per_player) {
  // Load the map.
  _map = std::make_unique<RTSMap>();
  _game_counter = -1;
  _max_num_units_per_player = max_num_units_per_player;
  Reset();
}

void GameEnv::Visualize() const {
  for (const auto& player : _players) {
    std::cout << player.PrintInfo() << std::endl;
  }
  // No FoW, everything.
  GameEnvAspect aspect(*this, INVALID, false);
  UnitIterator unit_iter(aspect, UnitIterator::ALL);
  while (!unit_iter.end()) {
    const Unit& u = *unit_iter;
    std::cout << u.PrintInfo() << std::endl;
    ++unit_iter;
  }
}

void GameEnv::ClearAllPlayers() {
  _players.clear();
}

void GameEnv::Reset() {
  _map->ClearMap();
  _next_unit_id = 0;
  _winner_id = INVALID;
  _terminated = false;
  _game_counter++;
  _units.clear();
  _bullets.clear();
  for (auto& player : _players) {
    player.ClearCache();
  }
  _game_status = ACTIVE_STATUS;
  for (size_t i = 0; i < _player_unit_counts.size(); ++i) {
    _player_unit_counts[i] = 0;
  }
}

void GameEnv::AddPlayer(const std::string& name) {
  // PlayerPrivilege pv,
  // PlayerType pt) {
  _players.emplace_back(*_map, name, _players.size());
  // _players.back().SetPrivilege(pv);
  // _players.back().SetType(pt);
  _player_unit_counts.push_back(0);
}

// void GameEnv::RemovePlayer() {
//   _players.pop_back();
// }

void GameEnv::SaveSnapshot(serializer::saver& saver) const {
  serializer::Save(saver, _next_unit_id);

  saver << _map;
  saver << _units;
  saver << _bullets;
  saver << _players;
  saver << _winner_id;
  saver << _terminated;
}

void GameEnv::LoadSnapshot(serializer::loader& loader) {
  serializer::Load(loader, _next_unit_id);

  loader >> _map;
  loader >> _units;
  loader >> _bullets;
  loader >> _players;
  loader >> _winner_id;
  loader >> _terminated;

  for (auto& player : _players) {
    player.ResetMap(_map.get());
  }
}

// Compute the hash code.
uint64_t GameEnv::CurrentHashCode() const {
  uint64_t code = 0;
  for (auto it = _units.begin(); it != _units.end(); ++it) {
    serializer::hash_combine(code, it->first);
    serializer::hash_combine(code, *it->second);
  }
  // Players.
  for (const auto& player : _players) {
    serializer::hash_combine(code, player);
  }
  return code;
}

bool GameEnv::AddUnit(Tick tick,
                      UnitType type,
                      const PointF& p,
                      PlayerId player_id,
                      Tick expiration) {
  // check if max_num_units is reached
  if (_max_num_units_per_player > 0 && type != RESOURCE) {
    // std::cout << "checking max num units,"
    //           << " player id: " << player_id
    //           << " count: " << _player_unit_counts.at(player_id)
    //           << std::endl;
    if (_player_unit_counts.at(player_id) >= _max_num_units_per_player) {
      return false;
    }
  }

  // check if there is any space.
  if (!_map->CanPass(p, INVALID, true, _gamedef.unit(type))) {
    return false;
  }
  // cout << "Actual adding unit." << endl;

  UnitId new_id = CombinePlayerId(_next_unit_id, player_id);

  Unit* new_unit = new Unit(
      tick, new_id, type, p, _gamedef.unit(type).GetProperty(), expiration);
  _units.insert(make_pair(new_id, std::unique_ptr<Unit>(new_unit)));
  _map->AddUnit(new_id, p);

  _next_unit_id++;
  if (type != RESOURCE) {
    _player_unit_counts.at(player_id) += 1;
  }
  return true;
}

bool GameEnv::RemoveUnit(const UnitId& id) {
  auto it = _units.find(id);
  if (it == _units.end()) {
    return false;
  }

  auto player_id = ExtractPlayerId(id);
  auto type = it->second->GetUnitType();

  _units.erase(it);
  _map->RemoveUnit(id);

  if (type != RESOURCE) {
    _player_unit_counts.at(player_id) -= 1;
  }
  return true;
}

void GameEnv::UpdateTemporaryUnits(Tick tick) {
  for (const auto& p : _units) {
    if (p.second->IsTemporary()) {
      p.second->UpdateBuildProgress(tick);
    }
  }
}

UnitId GameEnv::FindFirstTownHall(PlayerId player_id) const {
  // Find closest base. [TODO]: Not efficient here.
  for (auto it = _units.begin(); it != _units.end(); ++it) {
    const Unit* u = it->second.get();
    if (u->GetUnitType() == TOWN_HALL && u->GetPlayerId() == player_id) {
      return u->GetId();
    }
  }
  return INVALID;
}

UnitId GameEnv::FindClosestTownHall(PlayerId player_id, const PointF& p) const {
  const Unit* town_hall = FindClosestTownHallP(player_id, p);
  if (town_hall == nullptr) {
    return INVALID;
  }
  return town_hall->GetId();
}

const Unit* GameEnv::FindClosestTownHallP(PlayerId player_id,
                                          const PointF& p) const {
  const Unit* town_hall = nullptr;
  float closest = 1e10;
  for (auto it = _units.begin(); it != _units.end(); ++it) {
    const Unit* u = it->second.get();
    if (u->IsTemporary()) {
      continue;
    }
    if (u->GetUnitType() == TOWN_HALL && u->GetPlayerId() == player_id) {
      float dist_sqr = PointF::L2Sqr(p, u->GetPointF());
      if (dist_sqr < closest) {
        closest = dist_sqr;
        town_hall = u;
      }
    }
  }
  return town_hall;
}

UnitId GameEnv::FindClosestEnemy(PlayerId player_id,
                                 const PointF& p,
                                 float max_radius) const {
  UnitId id = INVALID;
  float closest = 1e10;
  for (auto it = _units.begin(); it != _units.end(); ++it) {
    const Unit* u = it->second.get();
    if (u->GetPlayerId() != player_id && u->GetUnitType() != RESOURCE) {
      float dist = PointF::L2Sqr(p, u->GetPointF());
      if (dist < max_radius * max_radius && dist < closest) {
        closest = dist;
        id = u->GetId();
      }
    }
  }
  return id;
}

PlayerId GameEnv::CheckTownHall() const {
  PlayerId last_player_has_town_hall = INVALID;
  for (auto it = _units.begin(); it != _units.end(); ++it) {
    const Unit* u = it->second.get();
    if (u->GetUnitType() == TOWN_HALL) {
      if (last_player_has_town_hall == INVALID) {
        last_player_has_town_hall = u->GetPlayerId();
      } else if (last_player_has_town_hall != u->GetPlayerId()) {
        // No winning.
        last_player_has_town_hall = INVALID;
        break;
      }
    }
  }
  return last_player_has_town_hall;
}

PlayerId GameEnv::CheckUnitsAndMoney() const {
  std::set<PlayerId> players_with_units, player_ids;
  for (auto& player : _players) {
    players_with_units.insert(player.GetId());
    player_ids.insert(player.GetId());
  }
  // TODO[hack]: Hack to remove coach during teamplay
  if (_players.size() == 3) {
    players_with_units.erase(1);
    player_ids.erase(1);
  }
  for (auto& it : _units) {
    const Unit* u = it.second.get();
    if (GameDef::IsUnitTypeBuilding(u->GetUnitType())) {
      continue;
    }
    if (u->GetUnitType() == RESOURCE || u->GetUnitType() == GUARD_TOWER) {
      continue;
    }
    players_with_units.erase(u->GetPlayerId());
  }
  // If only one player is without units
  if (players_with_units.size() != 1) {
    return INVALID;
  }
  const auto player_id = *players_with_units.begin();
  const auto& player = GetPlayer(player_id);
  if (!_gamedef.CanBuildAnyUnit(player.GetResource())) {
    // player_id has lost, return id of the opponents
    player_ids.erase(player_id);
    if (player_ids.size() == 1) {
      return *player_ids.begin();
    }
  }
  return INVALID;
}

bool GameEnv::FindEmptyPlaceNearby(const UnitTemplate& unit_def,
                                   const PointF& p,
                                   int l1_radius,
                                   const std::set<PointF>& place_taken,
                                   PointF* res_p) const {
  // Find an empty place by simple local grid search.
  const int margin = 2;
  const int cx = _map->GetXSize() / 2;
  const int cy = _map->GetYSize() / 2;
  int sx = p.x < cx ? -1 : 1;
  int sy = p.y < cy ? -1 : 1;

  for (int dx = -sx * l1_radius; dx != sx * l1_radius + sx; dx += sx) {
    for (int dy = -sy * l1_radius; dy != sy * l1_radius + sy; dy += sy) {
      PointF new_p(p.x + dx, p.y + dy);
      if (_map->CanPass(new_p, INVALID, true, unit_def) &&
          _map->IsIn(new_p, margin) &&
          place_taken.find(new_p) == place_taken.end()) {
        // It may not be a good strategy, though.
        *res_p = new_p;
        return true;
      }
    }
  }
  return false;
}

bool GameEnv::FindBuildPlaceNearby(const PointF& p,
                                   int l1_radius,
                                   PointF* res_p) const {
  // Find an empty place by simple local grid search.
  for (int dx = -l1_radius; dx <= l1_radius; dx++) {
    for (int dy = -l1_radius; dy <= l1_radius; dy++) {
      PointF new_p(p.x + dx, p.y + dy);
      if (_map->CanBuild(new_p, INVALID)) {
        *res_p = new_p;
        return true;
      }
    }
  }
  return false;
}

// given a set of units and a target point, a distance, find closest place to go
// to to maintain the distance. can be used by hit and run or scout.
bool GameEnv::FindClosestPlaceWithDistance(
    const Unit& u,
    const PointF& p,
    int dist,
    const std::vector<const Unit*>& units,
    PointF* res_p) const {
  const RTSMap& m = *_map;
  const UnitTemplate& unit_def = _gamedef.unit(u.GetUnitType());
  std::vector<Loc> distances(m.GetXSize() * m.GetYSize());
  std::vector<Loc> current;
  std::vector<Loc> nextloc;
  for (auto unit : units) {
    Loc loc = m.GetLoc(unit->GetPointF().ToCoord());
    distances[loc] = 0;
    current.push_back(loc);
  }
  const int dx[] = {1, 0, -1, 0};
  const int dy[] = {0, 1, 0, -1};
  for (int d = 1; d <= dist; d++) {
    for (Loc loc : current) {
      Coord c_curr = m.GetCoord(loc);
      for (size_t i = 0; i < sizeof(dx) / sizeof(int); ++i) {
        Coord next(c_curr.x + dx[i], c_curr.y + dy[i]);
        if (_map->CanPass(next, INVALID, false, unit_def) && _map->IsIn(next)) {
          Loc l_next = m.GetLoc(next);
          if (distances[l_next] == 0 || distances[l_next] > d) {
            nextloc.push_back(l_next);
            distances[l_next] = d;
          }
        }
      }
    }
    current = nextloc;
    nextloc.clear();
  }

  float closest = m.GetXSize() * m.GetYSize();
  bool found = false;
  for (Loc loc : current) {
    PointF pf = PointF(m.GetCoord(loc));
    float dist_sqr = PointF::L2Sqr(pf, p);
    if (closest > dist_sqr) {
      *res_p = pf;
      closest = dist_sqr;
      found = true;
    }
  }
  return found;
}

void GameEnv::Forward(CmdReceiver* receiver) {
  // Compute all bullets.
  if (!IsGameActive()) {
    return;
  }
  std::set<int> done_bullets;
  for (size_t i = 0; i < _bullets.size(); ++i) {
    CmdBPtr cmd = _bullets[i].Forward(*_map, _units);
    if (cmd.get() != nullptr) {
      // Note that this command is special. It should not be recorded in
      // the cmd_history.
      receiver->SetSaveToHistory(false);
      receiver->SendCmd(std::move(cmd));
      receiver->SetSaveToHistory(true);
    }
    if (_bullets[i].IsDead())
      done_bullets.insert(i);
  }

  // Remove bullets that are done.
  // Need to traverse in the reverse order.
  for (std::set<int>::reverse_iterator it = done_bullets.rbegin();
       it != done_bullets.rend();
       ++it) {
    unsigned int idx = *it;
    if (idx < _bullets.size() - 1) {
      std::swap(_bullets[idx], _bullets.back());
    }
    _bullets.pop_back();
  }
}

void GameEnv::ComputeFOW() {
  // Compute FoW.
  for (Player& p : _players) {
    p.ComputeFOW(_units);
  }
}

bool GameEnv::GenerateMap(int num_obstacles, int init_resource) {
  return _map->GenerateMap(
      GetRandomFunc(), num_obstacles, _players.size(), init_resource);
}

bool GameEnv::GenerateImpassable(int num_obstacles) {
  return _map->GenerateImpassable(GetRandomFunc(), num_obstacles);
}

std::string GameEnv::PrintDebugInfo() const {
  std::stringstream ss;
  ss << "Game #" << _game_counter << std::endl;
  for (const auto& player : _players) {
    ss << "Player " << player.GetId() << std::endl;
    ss << player.PrintHeuristicsCache() << std::endl;
  }

  ss << _map->Draw() << std::endl;
  ss << _map->PrintDebugInfo() << std::endl;
  return ss.str();
}

nlohmann::json GameEnv::LogMap2Json(PlayerId player_id) const {
  const Player& player = GetPlayer(player_id);
  const RTSMap& m = player.GetMap();

  nlohmann::json visibility;
  nlohmann::json terrain;

  for (int y = 0; y < m.GetYSize(); ++y) {
    for (int x = 0; x < m.GetXSize(); ++x) {
      Loc loc = m.GetLoc(Coord(x, y, 0));
      const Fog& f = player.GetFog(loc);
      Terrain t = FOG;

      visibility.push_back(f.GetVisibility());
      if (f.HasSeenTerrain()) {
        t = m(loc).type;
      }
      terrain.push_back(t);
    }
  }

  nlohmann::json map;
  map["visibility"] = visibility;
  map["terrain"] = terrain;
  return map;
}
