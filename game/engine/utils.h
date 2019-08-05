// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include "cmd.h"
#include "cmd_receiver.h"
#include "unit.h"
#include <chrono>
// #include "elf/comm/comm.h"
#include "common.h"
#include "const.h"
#include "game_env.h"

bool UnitHasCmd(UnitId id, const std::map<UnitId, CmdBPtr>& cmds);

// Some utility function to pick first from units in a group.
const Unit* PickFirst(const std::vector<const Unit*>& units,
                      const std::map<UnitId, CmdBPtr>& pendingCmds,
                      const CmdReceiver& receiver,
                      CmdType t);

const Unit* PickFirstIdle(const std::vector<const Unit*>& units,
                          const std::map<UnitId, CmdBPtr>& pendingCmds,
                          const CmdReceiver& receiver);

const Unit* PickIdleOrGather(const std::vector<const Unit*>& units,
                             const std::map<UnitId, CmdBPtr>& pendingCmds,
                             const CmdReceiver& receiver);

// check whether pending cmd is duplicated attack with current cmd
bool IsDuplicatedAttack(const CmdBPtr& pendingCmd,
                        UnitId id,
                        const CmdReceiver& receiver);

bool IsDuplicatedGather(const CmdBPtr& pendingCmd,
                        UnitId id,
                        const CmdReceiver& receiver);

bool IsAttackTemporaryBuilding(const CmdBPtr& pendingCmd, const GameEnv& env);

// check whether unit's current cmd is of certain type
bool InCmd(const CmdReceiver& receiver, const Unit& u, CmdType cmd);

bool InBuildingUnitType(const CmdReceiver& receiver,
                        const Unit& u,
                        UnitType utype,
                        bool* inBuild);

bool IsIdle(const CmdReceiver& receiver, const Unit& u);

bool IsGather(const CmdReceiver& receiver, const Unit& u);

bool IsIdleOrGather(const CmdReceiver& receiver, const Unit& u);

const Unit* closestUnit(const std::vector<const Unit*>& units,
                        const PointF& p,
                        float dsqrBound,
                        float* retDsqr = nullptr);

const Unit* closestUnitWithNoPendingCmd(
    const std::vector<const Unit*>& units,
    const PointF& p,
    const std::map<UnitId, CmdBPtr>& pendingCmds,
    const CmdReceiver& receiver,
    CmdType excludeCmd);

const Unit* closestUnitInCmd(const std::vector<const Unit*>& units,
                             const PointF& p,
                             const CmdReceiver& receiver,
                             CmdType cmdType);

std::vector<const Unit*> filterByDistance(const std::vector<const Unit*>& units,
                                          const PointF& p,
                                          float max_d);

template <typename T>
inline bool inVector(const std::vector<T>& vec, const T& val) {
  return std::find(vec.begin(), vec.end(), val) != vec.end();
}

inline PlayerId ExtractPlayerId(UnitId id) {
  // 24-30 encoding player id.
  return (id >> 24);
}

inline UnitId CombinePlayerId(UnitId raw_id, PlayerId player_id) {
  return (raw_id & 0xffffff) | (player_id << 24);
}

inline int get_time_microseconds_mod_by(
    int max = std::numeric_limits<int>::max()) {
  using namespace std::chrono;
  auto now = system_clock::now().time_since_epoch();
  microseconds mus = duration_cast<microseconds>(now);
  int reminder = mus.count() % max;
  return reminder;
}

bool RemoveTemporaryBuilding(const PointF& p,
                             UnitType build_type,
                             PlayerId player_id,
                             GameEnv* env,
                             CmdReceiver* receiver);

float micro_move(Tick tick,
                 const Unit& u,
                 const GameEnv& env,
                 const PointF& target,
                 CmdReceiver* receiver);

bool find_nearby_empty_place(const GameEnv& env,
                             const Unit& u,
                             const PointF& curr,
                             PointF* p_nearby);
