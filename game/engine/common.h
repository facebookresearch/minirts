// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include <cmath>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdarg.h>
#include <unordered_map>
#include <vector>

#include "custom_enum.h"
#include "serializer.h"

// All invalid numbers are -1
const int INVALID = -1;
const int MAX_GAME_LENGTH_EXCEEDED = -2;

enum GameResult { GAME_NORMAL = 0, GAME_END = 1, GAME_ERROR = 2 };
using Tick = int;

custom_enum(UnitType,
            INVALID_UNITTYPE = -1,
            /* Minirts unit types */
            RESOURCE = 0,
            PEASANT,
            SPEARMAN,
            SWORDMAN,
            CAVALRY,
            DRAGON,
            ARCHER,
            CATAPULT,
            /* buildings */
            BARRACK,
            BLACKSMITH,
            STABLE,
            WORKSHOP,
            AVIARY,
            ARCHERY,
            GUARD_TOWER,
            TOWN_HALL,
            NUM_MINIRTS_UNITTYPE);

custom_enum(GameStatus, ACTIVE_STATUS = 0, WAITING_STATUS, FROZEN_STATUS);

custom_enum(UnitAttr,
            INVALID_UNITATTR = -1,
            ATTR_NORMAL = 0,
            ATTR_INVULNERABLE,
            NUM_UNITATTR);

custom_enum(CDType,
            INVALID_CDTYPE = -1,
            CD_MOVE = 0,
            CD_ATTACK,
            CD_GATHER,
            CD_BUILD,
            NUM_COOLDOWN);

custom_enum(Terrain,
            INVALID_TERRAIN = -1,
            SOIL = 0,
            SAND,
            GRASS,
            ROCK,
            WATER,
            FOG,
            NUM_TERRAIN);

custom_enum(Visibility, VISIBLE = 0, SEEN, INVISIBLE, NUM_VISIBILITY);

custom_enum(BulletState,
            INVALID_BULLETSTATE = -1,
            BULLET_CREATE = 0,
            BULLET_CREATE1,
            BULLET_CREATE2,
            BULLET_READY,
            BULLET_EXPLODE1,
            BULLET_EXPLODE2,
            BULLET_EXPLODE3,
            BULLET_DONE);

// Map location (as integer)
using Loc = int;
using UnitId = int;
using PlayerId = int;
using Tick = int;

struct Coord {
  int x, y, z;
  Coord(int x, int y, int z = 0)
      : x(x)
      , y(y)
      , z(z) {
  }
  Coord()
      : x(0)
      , y(0)
      , z(0) {
  }

  Coord Left() const {
    Coord c(*this);
    c.x--;
    return c;
  }
  Coord Right() const {
    Coord c(*this);
    c.x++;
    return c;
  }
  Coord Up() const {
    Coord c(*this);
    c.y--;
    return c;
  }
  Coord Down() const {
    Coord c(*this);
    c.y++;
    return c;
  }

  friend std::ostream& operator<<(std::ostream& oo, const Coord& p) {
    oo << p.x << " " << p.y;
    return oo;
  }

  friend std::istream& operator>>(std::istream& ii, Coord& p) {
    ii >> p.x >> p.y;
    p.z = 0;
    return ii;
  }

  bool operator==(const Coord& other) const {
    return x == other.x && y == other.y && z == other.z;
  }

  bool operator!=(const Coord& other) const {
    return !(*this == other);
  }

  SERIALIZER(Coord, x, y, z);
  HASH(Coord, x, y, z);
};

STD_HASH(Coord);

// Seems that this should be larger than \sqrt{2}/2/2 = 0.35355339059
const float kUnitRadius = 0.36;

struct PointF {
  float x, y;
  PointF(float x, float y)
      : x(x)
      , y(y) {
  }
  PointF() {
    SetInvalid();
  }
  PointF(const Coord& c)
      : x(c.x)
      , y(c.y) {
  }

  bool operator<(const PointF& other) const {
    if (x < other.x) {
      return true;
    } else if (x == other.x) {
      return y < other.y;
    } else {
      return false;
    }
  }

  Coord ToCoord() const {
    return Coord((int)(x + 0.5), (int)(y + 0.5));
  }

  PointF self() const {
    return *this;
  }

  PointF Left() const {
    PointF c(*this);
    c.x -= 1.0;
    return c;
  }
  PointF Right() const {
    PointF c(*this);
    c.x += 1.0;
    return c;
  }
  PointF Up() const {
    PointF c(*this);
    c.y -= 1.0;
    return c;
  }
  PointF Down() const {
    PointF c(*this);
    c.y += 1.0;
    return c;
  }

  PointF LT() const {
    PointF c(*this);
    c.x -= 1.0;
    c.y -= 1.0;
    return c;
  }
  PointF LB() const {
    PointF c(*this);
    c.x -= 1.0;
    c.y += 1.0;
    return c;
  }
  PointF RT() const {
    PointF c(*this);
    c.x += 1.0;
    c.y -= 1.0;
    return c;
  }
  PointF RB() const {
    PointF c(*this);
    c.x += 1.0;
    c.y += 1.0;
    return c;
  }

  PointF LL() const {
    PointF c(*this);
    c.x -= 2.0;
    return c;
  }
  PointF RR() const {
    PointF c(*this);
    c.x += 2.0;
    return c;
  }
  PointF TT() const {
    PointF c(*this);
    c.y -= 2.0;
    return c;
  }
  PointF BB() const {
    PointF c(*this);
    c.y += 2.0;
    return c;
  }

  bool IsInvalid() const {
    return x < -1e17 || y < -1e17;
  }
  bool IsValid() const {
    return !IsInvalid();
  }
  void SetInvalid() {
    x = -1e18;
    y = -1e18;
  }

  PointF CCW90() const {
    return PointF(-y, x);
  }
  PointF CW90() const {
    return PointF(y, -x);
  }
  PointF Negate() const {
    return PointF(-y, -x);
  }

  const PointF& Trunc(float mag) {
    float l = std::sqrt(x * x + y * y);
    if (l > mag) {
      x *= mag / l;
      y *= mag / l;
    }
    return *this;
  }

  PointF& operator+=(const PointF& p) {
    x += p.x;
    y += p.y;
    return *this;
  }

  PointF& operator-=(const PointF& p) {
    x -= p.x;
    y -= p.y;
    return *this;
  }

  PointF& operator*=(float s) {
    x *= s;
    y *= s;
    return *this;
  }

  PointF& operator/=(float s) {
    x /= s;
    y /= s;
    return *this;
  }

  friend PointF operator-(const PointF& p1, const PointF& p2) {
    return PointF(p1.x - p2.x, p1.y - p2.y);
  }

  bool IsIn(const PointF& left_top, const PointF& right_down) const {
    return left_top.x <= x && x <= right_down.x && left_top.y <= y &&
           y <= right_down.y;
  }

  static bool Diff(const PointF& p1, const PointF& p2, PointF* d) {
    if (p1.IsInvalid() || p2.IsInvalid())
      return false;
    d->x = p1.x - p2.x;
    d->y = p1.y - p2.y;
    return true;
  }

  friend std::ostream& operator<<(std::ostream& oo, const PointF& p) {
    oo << p.x << " " << p.y;
    return oo;
  }

  friend std::istream& operator>>(std::istream& ii, PointF& p) {
    ii >> p.x >> p.y;
    return ii;
  }

  static float L2Sqr(const PointF& p1, const PointF& p2) {
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    return dx * dx + dy * dy;
  }

  std::string info() const {
    return std::to_string(x) + ", " + std::to_string(y);
  }

  void SetIntXY(int xx, int yy) {
    x = xx;
    y = yy;
  }

  SERIALIZER(PointF, x, y);
  HASH(PointF, x, y);
};

STD_HASH(PointF);

struct RegionF {
  PointF c;
  float r;
};

// Function that make an arbitrary list into a string, separated by space.
template <typename T> void _make_string(std::stringstream& oo, T v) {
  oo << v;
}

template <typename T, typename... Args>
void _make_string(std::stringstream& oo, T v, Args... args) {
  oo << v << " ";
  _make_string(oo, args...);
}

template <typename... Args> std::string make_string(Args... args) {
  std::stringstream s;
  _make_string(s, args...);
  return s.str();
}

struct Cooldown {
  Tick _last;
  int _cd;
  void Set(int cd_val) {
    _last = 0;
    _cd = cd_val;
  }
  void Start(Tick tick) {
    _last = tick;
  }
  bool Passed(Tick tick) const {
    return tick - _last >= _cd;
  }
  std::string PrintInfo(Tick tick = INVALID) const {
    std::stringstream ss;
    ss << "_last: " << _last << " cd: " << _cd;
    if (tick != INVALID)
      ss << "tick: " << tick << ", tick-_last: " << tick - _last;
    return ss.str();
  }

  SERIALIZER(Cooldown, _last, _cd);
  HASH(Cooldown, _cd, _last);
};

struct BuildSkill {
  int _unit_type;
  std::string _hotkey;

  UnitType GetUnitType() const {
    return static_cast<UnitType>(_unit_type);
  }
  std::string GetHotKey() const {
    return _hotkey;
  }
};

STD_HASH(Cooldown);
