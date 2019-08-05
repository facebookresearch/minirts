// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include "common.h"
#include "gamedef.h"
#include "locality_search.h"
#include <functional>
#include <vector>

struct MapSlot {
  // three layers, terrian, ground and air.
  Terrain type;
  int height;

  // Pre-computed distance map for fog of war computation (and path-planning).
  // sorted nearest neighbor after considering the map structure.
  std::vector<Loc> _nns;

  // Default constructor
  MapSlot()
      : type(GRASS)
      , height(0) {
  }

  SERIALIZER(MapSlot, type, height, _nns);
};

struct PlayerMapInfo {
  PlayerId player_id;
  Coord base_coord;
  Coord resource_coord;
  int initial_resource;

  SERIALIZER(
      PlayerMapInfo, player_id, base_coord, resource_coord, initial_resource);
};

// Map properties.
// Map location is an integer.
class RTSMap {
 private:
  std::vector<MapSlot> _map;

  // Size of the map.
  int _m, _n, _level;

  // Initial players' info
  std::vector<PlayerMapInfo> _infos;

  // Locality search.
  LocalitySearch<UnitId> _locality;

  void load_default_map();
  void precompute_all_pair_distances();

  bool find_two_nearby_empty_slots(const std::function<uint16_t(int)>& f,
                                   int* x1,
                                   int* y1,
                                   int* x2,
                                   int* y2,
                                   int i) const;

  bool can_pass(const PointF& p,
                const Coord& c,
                UnitId id_exclude,
                bool seen_location,
                const UnitTemplate& unit_def,
                bool check_locality) const {
    if (!IsIn(c))
      return false;

    // Only check the observed part of the map
    if (seen_location) {
      Loc loc = GetLoc(c);
      const MapSlot& s = _map[loc];
      if (!unit_def.CanMoveOver(s.type))
        return false;
    }

    // [TODO] Add object radius here.
    if (check_locality)
      return _locality.IsEmpty(p, kUnitRadius, id_exclude);
    else
      return true;
  }

 public:
  // Load map from a file.
  RTSMap()
      : _m(0)
      , _n(0) {
  }

  bool GenerateMap(const std::function<uint16_t(int)>& f,
                   int nImpassable,
                   int num_player,
                   int init_resource);

  bool LoadMap(const std::string& filename);

  bool GenerateImpassable(const std::function<uint16_t(int)>& f,
                          int nImpassable);

  void InitMap(int m, int n, int level);

  void ResetIntermediates();

  const std::vector<PlayerMapInfo>& GetPlayerMapInfo() const {
    return _infos;
  }
  void ClearMap() {
    _infos.clear();
    _locality.Clear();
  }

  void AddPlayer(int player_id,
                 int base_x,
                 int base_y,
                 int resource_x,
                 int resource_y,
                 int init_resource);

  const MapSlot& operator()(const Loc& loc) const {
    return _map[loc];
  }
  MapSlot& operator()(const Loc& loc) {
    return _map[loc];
  }

  // Lua get/set
  int GetSlotType(int x, int y, int z = 0) {
    if (IsIn(x, y)) {
      return static_cast<int>(_map[GetLoc(Coord(x, y, z))].type);
    }
    return static_cast<int>(INVALID_TERRAIN);
  }

  void SetSlotType(int terrain_type, int x, int y, int z = 0) {
    if (IsIn(x, y)) {
      _map[GetLoc(Coord(x, y, z))].type = static_cast<Terrain>(terrain_type);
    }
  }

  int GetXSize() const {
    return _m;
  }
  int GetYSize() const {
    return _n;
  }
  void SetXSize(int x) {
    _m = x;
  }
  void SetYSize(int y) {
    _n = y;
  }

  int GetPlaneSize() const {
    return _m * _n;
  }

  // TODO: move this to game_TD
  bool CanBuild(const PointF& p, UnitId id_exclude) const {
    Coord c = p.ToCoord();
    if (!IsIn(c)) {
      return false;
    }

    // [TODO] Add object radius here.
    return _locality.IsEmpty(p, kUnitRadius, id_exclude);
  }

  bool CanPass(const PointF& p,
               UnitId id_exclude,
               bool seen_location = false,
               const UnitTemplate& unit_def = UnitTemplate(),
               bool check_locality = true) const {
    Coord c = p.ToCoord();
    return can_pass(p, c, id_exclude, seen_location, unit_def, check_locality);
  }

  bool CanPass(const Coord& c,
               UnitId id_exclude,
               bool seen_location = false,
               const UnitTemplate& unit_def = UnitTemplate(),
               bool check_locality = true) const {
    PointF p(c.x, c.y);
    return can_pass(p, c, id_exclude, seen_location, unit_def, check_locality);
  }

  bool IsLinePassable(const PointF& s, const PointF& t) const {
    LineResult result;
    UnitId block_id = INVALID;
    return _locality.LinePassable(s, t, kUnitRadius, 1e-4, &block_id, &result);
  }

  bool CanSee(const Coord& s, const Coord& t) const;

  // Move a unit to next_loc;
  bool MoveUnit(const UnitId& id, const PointF& new_loc);
  bool AddUnit(const UnitId& id, const PointF& new_loc);
  bool RemoveUnit(const UnitId& id);

  // Coord transfer.
  std::string PrintCoord(Loc loc) const;
  Coord GetCoord(Loc loc) const;
  Loc GetLoc(const Coord& c) const;
  Loc GetLoc2d(int x, int y) const;

  bool IsIn(Loc loc) const {
    return loc >= 0 && loc < _m * _n * _level;
  }
  bool IsIn(int x, int y) const {
    return x >= 0 && x < _m && y >= 0 && y < _n;
  }
  bool IsIn(const Coord& c, int margin = 0) const {
    return c.x >= margin && c.x < _m - margin && c.y >= margin &&
           c.y < _n - margin && c.z >= 0 && c.z < _level;
  }
  bool IsIn(const PointF& p, int margin = 0) const {
    return IsIn(p.ToCoord(), margin);
  }

  UnitId GetClosestUnitId(const PointF& p, float max_r = 1e38) const;
  std::set<UnitId> GetUnitIdInRegion(const PointF& left_top,
                                     const PointF& right_bottom) const;

  // Draw the map
  std::string Draw() const;

  std::string PrintDebugInfo() const;

  SERIALIZER(RTSMap, _m, _n, _level, _map, _infos, _locality);
};
