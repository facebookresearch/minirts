// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#include "utils.h"
#include "cmd.gen.h"
#include "cmd.h"
#include "cmd_specific.gen.h"

bool UnitHasCmd(UnitId id, const std::map<UnitId, CmdBPtr>& cmds) {
  auto it = cmds.find(id);
  return (it != cmds.end());
}

// Some utility function to pick first from units in a group.
const Unit* PickFirst(const std::vector<const Unit*>& units,
                      const std::map<UnitId, CmdBPtr>& pendingCmds,
                      const CmdReceiver& receiver,
                      CmdType t) {
  for (const auto* u : units) {
    if (UnitHasCmd(u->GetId(), pendingCmds)) {
      continue;
    }

    const CmdDurative* cmd = receiver.GetUnitDurativeCmd(u->GetId());
    if ((t == INVALID_CMD && cmd == nullptr) ||
        (cmd != nullptr && cmd->type() == t)) {
      return u;
    }
  }
  return nullptr;
}

const Unit* PickFirstIdle(const std::vector<const Unit*>& units,
                          const std::map<UnitId, CmdBPtr>& pendingCmds,
                          const CmdReceiver& receiver) {
  return PickFirst(units, pendingCmds, receiver, INVALID_CMD);
}

const Unit* PickIdleOrGather(const std::vector<const Unit*>& units,
                             const std::map<UnitId, CmdBPtr>& pendingCmds,
                             const CmdReceiver& receiver) {
  const Unit* gather = nullptr;
  for (const auto* u : units) {
    if (UnitHasCmd(u->GetId(), pendingCmds)) {
      continue;
    }

    const CmdDurative* cmd = receiver.GetUnitDurativeCmd(u->GetId());
    if (cmd == nullptr) {
      return u;
    }
    if (cmd->type() == GATHER && gather == nullptr) {
      gather = u;
    }
  }
  return gather;
}

bool IsDuplicatedAttack(const CmdBPtr& pendingCmd,
                        UnitId id,
                        const CmdReceiver& receiver) {
  if (pendingCmd->type() != ATTACK) {
    return false;
  }

  const CmdDurative* currentCmd = receiver.GetUnitDurativeCmd(id);
  if (currentCmd == nullptr || currentCmd->type() != ATTACK) {
    return false;
  }

  const CmdAttack* currentAtt = dynamic_cast<const CmdAttack*>(currentCmd);
  const CmdAttack* pendingAtt =
      dynamic_cast<const CmdAttack*>(pendingCmd.get());
  return currentAtt->target() == pendingAtt->target();
}

bool IsDuplicatedGather(const CmdBPtr& pendingCmd,
                        UnitId id,
                        const CmdReceiver& receiver) {
  if (pendingCmd->type() != GATHER) {
    return false;
  }

  const CmdDurative* currentCmd = receiver.GetUnitDurativeCmd(id);
  if (currentCmd == nullptr || currentCmd->type() != GATHER) {
    return false;
  }

  const CmdGather* current = dynamic_cast<const CmdGather*>(currentCmd);
  const CmdGather* pending = dynamic_cast<const CmdGather*>(pendingCmd.get());
  bool duplicated = (current->resource() == pending->resource() &&
                     current->town_hall() == pending->town_hall());
  return duplicated;
}

bool IsAttackTemporaryBuilding(const CmdBPtr& pendingCmd, const GameEnv& env) {
  if (pendingCmd->type() != ATTACK) {
    return false;
  }
  const CmdAttack* attack = dynamic_cast<const CmdAttack*>(pendingCmd.get());
  const Unit* target = env.GetUnit(attack->target());
  if (target == nullptr) {
    return false;
  }
  return target->IsTemporary();
}

bool InCmd(const CmdReceiver& receiver, const Unit& u, CmdType cmd) {
  const CmdDurative* c = receiver.GetUnitDurativeCmd(u.GetId());
  return (c != nullptr && c->type() == cmd);
}

bool InBuildingUnitType(const CmdReceiver& receiver,
                        const Unit& u,
                        UnitType utype,
                        bool* inBuild) {
  const CmdDurative* c = receiver.GetUnitDurativeCmd(u.GetId());
  if (c == nullptr || c->type() != BUILD) {
    (*inBuild) = false;
    return false;
  }

  (*inBuild) = true;
  const CmdBuild* build = dynamic_cast<const CmdBuild*>(c);
  if (build->build_type() == utype) {
    return true;
  }
  return false;
}

bool IsIdle(const CmdReceiver& receiver, const Unit& u) {
  return receiver.GetUnitDurativeCmd(u.GetId()) == nullptr;
}

bool IsGather(const CmdReceiver& receiver, const Unit& u) {
  auto cmd = receiver.GetUnitDurativeCmd(u.GetId());
  return (cmd != nullptr && cmd->type() == GATHER);
}

bool IsIdleOrGather(const CmdReceiver& receiver, const Unit& u) {
  auto cmd = receiver.GetUnitDurativeCmd(u.GetId());
  return (cmd == nullptr || cmd->type() == GATHER);
}

const Unit* closestUnit(const std::vector<const Unit*>& units,
                        const PointF& p,
                        float dsqrBound,
                        float* retDsqr) {
  const Unit* closestUnit = nullptr;
  for (const auto* u : units) {
    float dsqr = PointF::L2Sqr(u->GetPointF(), p);
    if (dsqr < dsqrBound) {
      dsqrBound = dsqr;
      closestUnit = u;
    }
  }
  if (retDsqr != nullptr) {
    *retDsqr = dsqrBound;
  }
  return closestUnit;
}

const Unit* closestUnitWithNoPendingCmd(
    const std::vector<const Unit*>& units,
    const PointF& p,
    const std::map<UnitId, CmdBPtr>& pendingCmds,
    const CmdReceiver& receiver,
    CmdType excludeCmd) {
  const Unit* closestUnit = nullptr;
  float minDsqr = 0;
  for (const auto* u : units) {
    if (InCmd(receiver, *u, excludeCmd)) {
      continue;
    }
    if (UnitHasCmd(u->GetId(), pendingCmds)) {
      continue;
    }

    float dsqr = PointF::L2Sqr(u->GetPointF(), p);
    if (closestUnit == nullptr || dsqr < minDsqr) {
      minDsqr = dsqr;
      closestUnit = u;
    }
  }
  return closestUnit;
}

const Unit* closestUnitInCmd(const std::vector<const Unit*>& units,
                             const PointF& p,
                             const CmdReceiver& receiver,
                             CmdType cmdType) {
  const Unit* closestUnit = nullptr;
  float minDsqr = 0;
  for (const auto* u : units) {
    if (!InCmd(receiver, *u, cmdType)) {
      continue;
    }
    float dsqr = PointF::L2Sqr(u->GetPointF(), p);
    if (closestUnit == nullptr || dsqr < minDsqr) {
      minDsqr = dsqr;
      closestUnit = u;
    }
  }
  return closestUnit;
}

std::vector<const Unit*> filterByDistance(const std::vector<const Unit*>& units,
                                          const PointF& p,
                                          float max_d) {
  float max_dsqr = max_d * max_d;
  std::vector<const Unit*> closeUnits;
  for (auto u : units) {
    float dsqr = PointF::L2Sqr(u->GetPointF(), p);
    if (dsqr < max_dsqr) {
      closeUnits.emplace_back(u);
    }
  }
  return closeUnits;
}

bool RemoveTemporaryBuilding(const PointF& p,
                             UnitType build_type,
                             PlayerId player_id,
                             GameEnv* env,
                             CmdReceiver* receiver) {
  // find temporary building and expire it
  for (const auto& pair : env->GetUnits()) {
    const auto& unit = pair.second;
    if (unit->IsTemporary() && unit->GetUnitType() == build_type &&
        (INVALID == player_id || unit->GetPlayerId() == player_id) &&
        PointF::L2Sqr(unit->GetPointF(), p) < 1e-4) {
      const auto unit_id = unit->GetId();
      if (env->RemoveUnit(unit_id)) {
        receiver->FinishDurativeCmd(unit_id);
        return true;
      }
    }
  }
  return false;
}

#define MT_OK 0
#define MT_TARGET_INVALID 1
#define MT_ARRIVED 2
#define MT_CANNOT_MOVE 3

int move_toward(const RTSMap& m,
                const UnitTemplate& unit_def,
                float speed,
                const UnitId& id,
                const PointF& curr,
                const PointF& target,
                PointF* move) {
  // Given curr location, move towards the target.
  PointF diff;

  if (!PointF::Diff(target, curr, &diff)) {
    return MT_TARGET_INVALID;
  }
  if (std::abs(diff.x) < kDistEps && std::abs(diff.y) < kDistEps) {
    return MT_ARRIVED;
  }

  // bool movable = false;
  while (true) {
    diff.Trunc(speed);
    PointF next_p(curr);
    next_p += diff;

    bool movable = m.CanPass(next_p, id, true, unit_def);
    // cout << "MoveToward [" << id << "]: Try straight: " << next_p << "
    // movable: " << movable << endl;

    if (!movable) {
      next_p = curr;
      next_p += diff.CCW90();
      movable = m.CanPass(next_p, id, true, unit_def);
      // cout << "MoveToward [" << id << "]: Try CCW: " << next_p << " movable:
      // " << movable << endl;
    }
    if (!movable) {
      next_p = curr;
      next_p += diff.CW90();
      movable = m.CanPass(next_p, id, true, unit_def);
      // cout << "MoveToward [" << id << "]: Try CW: " << next_p << " movable: "
      // << movable << endl;
    }

    // If we still cannot move, then we reduce the speed.
    if (movable) {
      *move = next_p;
      return MT_OK;
    } else {
      return MT_CANNOT_MOVE;
    }
    /*
    speed /= 1.2;
    // If the move speed is too slow, then we skip.
    if (speed < 0.005) break;
    */
  }
}

float micro_move(Tick tick,
                 const Unit& u,
                 const GameEnv& env,
                 const PointF& target,
                 CmdReceiver* receiver) {
  const RTSMap& m = env.GetMap();
  const PointF& curr = u.GetPointF();
  const Player& player = env.GetPlayer(u.GetPlayerId());
  const UnitTemplate& unit_def = env.GetGameDef().unit(u.GetUnitType());

  // if (target.x < 0 || target.y < 0) {
  //   std::cout << "Micro_move: Current: " << curr
  //             << " Target: " << target << std::endl;
  // }

  float dist_sqr = PointF::L2Sqr(target, curr);
  const UnitProperty& property = u.GetProperty();

  static const int kMaxPlanningIteration = 1000;

  // Move to a target. Ideally we should do path-finding, for now just follow L1
  // distance.
  if (property.CD(CD_MOVE).Passed(tick)) {
    PointF move;

    float dist_sqr = PointF::L2Sqr(curr, target);
    PointF waypoint = target;
    bool planning_success = false;

    if (dist_sqr > 1.0) {
      // Do path planning.
      Coord first_block;
      float est_dist;
      planning_success = player.PathPlanning(tick,
                                             u.GetId(),
                                             unit_def,
                                             curr,
                                             target,
                                             kMaxPlanningIteration,
                                             receiver->GetPathPlanningVerbose(),
                                             &first_block,
                                             &est_dist);
      if (planning_success && first_block.x >= 0 && first_block.y >= 0) {
        waypoint.x = first_block.x;
        waypoint.y = first_block.y;
      }
    }
    // cout << "micro_move: (" << curr << ") -> (" << waypoint << ") planning: "
    //      << planning_success << endl;
    int ret = move_toward(
        m, unit_def, property._speed, u.GetId(), curr, waypoint, &move);
    if (ret == MT_OK) {
      // if (move.x < 0 || move.y < 0) {
      //   std::cout << "micro move: " << move.x << ", " << move.y << std::endl;
      // }
      // Set actual move.
      receiver->SendCmd(CmdIPtr(new CmdTacticalMove(u.GetId(), move)));
    } else if (ret == MT_CANNOT_MOVE) {
      // Unable to move. This is usually due to PathPlanning issues.
      // Too many such commands will leads to early termination of game.
      // [TODO]: Make PathPlanning better.
      receiver->GetGameStats().RecordFailedMove(tick, 1.0);
    }
  }
  return dist_sqr;
}

bool find_nearby_empty_place(const GameEnv& env,
                             const Unit& u,
                             const PointF& curr,
                             PointF* p_nearby) {
  const RTSMap& m = env.GetMap();
  const UnitTemplate& unit_def = env.GetGameDef().unit(u.GetUnitType());

  PointF nn;
  nn = curr.Left();
  if (m.CanPass(nn, INVALID, true, unit_def)) {
    *p_nearby = nn;
    return true;
  }
  nn = curr.Right();
  if (m.CanPass(nn, INVALID, true, unit_def)) {
    *p_nearby = nn;
    return true;
  }
  nn = curr.Up();
  if (m.CanPass(nn, INVALID, true, unit_def)) {
    *p_nearby = nn;
    return true;
  }
  nn = curr.Down();
  if (m.CanPass(nn, INVALID, true, unit_def)) {
    *p_nearby = nn;
    return true;
  }

  nn = curr.LT();
  if (m.CanPass(nn, INVALID, true, unit_def)) {
    *p_nearby = nn;
    return true;
  }
  nn = curr.LB();
  if (m.CanPass(nn, INVALID, true, unit_def)) {
    *p_nearby = nn;
    return true;
  }
  nn = curr.RT();
  if (m.CanPass(nn, INVALID, true, unit_def)) {
    *p_nearby = nn;
    return true;
  }
  nn = curr.RB();
  if (m.CanPass(nn, INVALID, true, unit_def)) {
    *p_nearby = nn;
    return true;
  }

  nn = curr.LL();
  if (m.CanPass(nn, INVALID, true, unit_def)) {
    *p_nearby = nn;
    return true;
  }
  nn = curr.RR();
  if (m.CanPass(nn, INVALID, true, unit_def)) {
    *p_nearby = nn;
    return true;
  }
  nn = curr.TT();
  if (m.CanPass(nn, INVALID, true, unit_def)) {
    *p_nearby = nn;
    return true;
  }
  nn = curr.BB();
  if (m.CanPass(nn, INVALID, true, unit_def)) {
    *p_nearby = nn;
    return true;
  }

  // std::cout << "WARNING: no avail space at all" << std::endl;
  return false;
}
