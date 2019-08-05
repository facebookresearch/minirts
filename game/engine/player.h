// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include "cmd.h"
#include "engine/unit.h"
#include "gamedef.h"
#include "map.h"
#include <queue>

class Unit;

struct Instruction {
  std::string _text;
  Tick _tick_issued = -1;
  Tick _tick_finished = -1;
  bool _done = false;
  bool _warn = false;
};

class Fog {
 private:
  // Fog level: 0 no fog, 100 completely invisible.
  bool _visible = false;
  bool _has_seen_terrain = false;
  std::set<Unit> _prev_seen_units;

 public:
  void SetClear() {
    _visible = true;
    _has_seen_terrain = true;
    _prev_seen_units.clear();
  }

  void SaveUnit(const Unit& u) {
    if (!GameDef::IsUnitTypeBuilding(u.GetUnitType())) {
      return;
    }

    auto finder = _prev_seen_units.find(u);
    assert(finder == _prev_seen_units.end());
    _prev_seen_units.insert(u);
  }

  bool IsSavedUnit(const Unit& u) const {
    return _prev_seen_units.count(u) > 0;
  }

  bool IsVisible() const {
    return _visible;
  }

  void MakeInvisible() {
    _visible = false;
  }

  Visibility GetVisibility() const {
    if (_visible) {
      return VISIBLE;
    } else if (_has_seen_terrain) {
      return SEEN;
    } else {
      return INVISIBLE;
    }
  }

  bool HasSeenTerrain() const {
    return _has_seen_terrain;
  }

  void ResetFog() {
    _visible = false;
    _has_seen_terrain = false;
    _prev_seen_units.clear();
  }

  const std::set<Unit>& seen_units() const {
    return _prev_seen_units;
  }

  SERIALIZER(Fog, _visible);
};

// // PlayerPrivilege, Normal player only see within the Fog of War.
// // KnowAll Player knows everything and can attack objects outside its
// // FOW.
// custom_enum(PlayerPrivilege, PV_NORMAL = 0, PV_KNOW_ALL);

// // Type of player
// custom_enum(PlayerType, PT_PLAYER = 0, PT_COACH);

class Player {
 private:
  const RTSMap* _map;

  // Player information.
  PlayerId _player_id;
  std::string _name;

  // // Type of players, different player could have different privileges.
  // PlayerPrivilege _privilege;
  // PlayerType _type;

  // Used in score computation.
  // How many resources the player have.
  int _resource;

  // Current fog of war. This containers have the same size as the map.
  std::vector<Fog> _fogs;

  // Instruction that the player needs to follow
  std::vector<Instruction> _instructions;

  // Heuristic function for path-planning.
  // Loc x Loc -> min distance (in discrete space).
  // If the key is not in _heuristics, then by default it is l2 distance.
  mutable std::map<std::pair<Loc, Loc>, float> _heuristics;

  // Cache for path planning. If the cache is too old, it will recompute.
  // Loc == INVALID: cannot pass / passable by a straight line (In this case, we
  // return first_block = -1.
  mutable std::map<std::pair<Loc, Loc>, std::pair<Tick, Loc>> _cache;

  struct Item {
    float g;
    float h;
    float cost;
    Loc loc;
    Loc loc_from;
    Item(float g, float h, const Loc& loc, const Loc& loc_from)
        : g(g)
        , h(h)
        , loc(loc)
        , loc_from(loc_from) {
      cost = g + h;
    }

    friend bool operator<(const Item& m1, const Item& m2) {
      if (m1.cost > m2.cost)
        return true;
      if (m1.cost < m2.cost)
        return false;
      if (m1.g > m2.g)
        return true;
      if (m1.g < m2.g)
        return false;
      if (m1.loc < m2.loc)
        return true;
      if (m1.loc > m2.loc)
        return false;
      if (m1.loc_from < m2.loc_from)
        return true;
      if (m1.loc_from > m2.loc_from)
        return false;
      return true;
    }

    std::string PrintInfo(const RTSMap& m) const {
      std::stringstream ss;
      ss << "cost: " << cost << " loc: (" << m.GetCoord(loc) << ") g: " << g
         << " h: " << h << " from: (" << m.GetCoord(loc_from) << ")";
      return ss.str();
    }
  };

  Loc _filter_with_fow(const Unit& u, bool check_saved_units) const;

  bool line_passable(const UnitTemplate& unit_def,
                     UnitId id,
                     const PointF& curr,
                     const PointF& target) const;

  float get_line_dist(const Loc& p1, const Loc& p2) const;

  // Update the heuristic value.
  void update_heuristic(const Loc& p1, const Loc& p2, float value) const;

  // Get the heuristic distance from p1 to p2.
  float get_path_dist_heuristic(const Loc& p1, const Loc& p2) const;

 public:
  Player()
      : _map(nullptr)
      , _player_id(INVALID)
      ,
      // _privilege(PV_NORMAL),
      // _type(PT_PLAYER),
      _resource(0) {
  }

  Player(const RTSMap& m, const std::string& name, int player_id)
      : _map(&m)
      , _player_id(player_id)
      , _name(name)
      ,
      // _privilege(PV_NORMAL),
      // _type(PT_PLAYER),
      _resource(0) {
  }

  const RTSMap& GetMap() const {
    return *_map;
  }

  const RTSMap* ResetMap(const RTSMap* new_map) {
    auto tmp = _map;
    _map = new_map;
    return tmp;
  }

  PlayerId GetId() const {
    return _player_id;
  }

  const std::string& GetName() const {
    return _name;
  }

  int GetResource() const {
    return _resource;
  }

  void CreateFog();
  std::string Draw() const;

  void ClearFogInSight(const Loc& loc, int range, const RTSMap& map);

  void ComputeFOW(const Units& units);

  bool FilterWithFOW(const Unit& u, bool check_saved_units) const;

  float GetDistanceSquared(const PointF& p, const Coord& c) const {
    float dx = p.x - c.x;
    float dy = p.y - c.y;
    return dx * dx + dy * dy;
  }

  // It will change _heuristics internally.
  bool PathPlanning(Tick tick,
                    UnitId id,
                    const UnitTemplate& unit_def,
                    const PointF& curr,
                    const PointF& target,
                    int max_iteration,
                    bool verbose,
                    Coord* first_block,
                    float* est_dist) const;

  // void SetPrivilege(PlayerPrivilege new_pv) {
  //   _privilege = new_pv;
  // }

  // PlayerPrivilege GetPrivilege() const {
  //   return _privilege;
  // }

  // void SetType(PlayerType new_pt) {
  //   _type = new_pt;
  // }

  // PlayerType GetType() const {
  //   return _type;
  // }

  int ChangeResource(int delta) {
    _resource += delta;
    // cout << "Base resource = " << _resource << endl;
    return _resource;
  }

  std::string DrawStats() const {
    return make_string("p", _player_id, _resource);
  }

  void ClearCache() {
    _heuristics.clear();
    _cache.clear();
    _resource = 0;
    for (auto& fog : _fogs) {
      fog.ResetFog();
    }
  }

  const Fog& GetFog(Loc loc) const {
    return _fogs.at(loc);
  }

  std::string PrintInfo() const;
  std::string PrintHeuristicsCache() const;

  void IssueInstruction(Tick tick, const std::string& instruction);
  void FinishInstruction(Tick tick);
  void WarnInstruction();

  const std::vector<Instruction>& GetInstructions() const {
    // std::cout << ">>>>>>>>>" << "get instructions called for player: "
    //           << _player_id << std::endl;
    // for (auto& inst : _instructions) {
    //   std::cout << inst._text << std::endl;
    // }
    return _instructions;
  }

  SERIALIZER(Player, _player_id, _resource, _fogs, _heuristics, _cache);
  HASH(Player, _player_id, _resource);
};

STD_HASH(Player);
