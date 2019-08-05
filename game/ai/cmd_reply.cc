// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#include "engine/rule_actor.h"
#include "ai/cmd_reply.h"

bool writeIdle(const GameEnv&, const Unit*, const CmdReceiver&, bool) {
  // assert(u != nullptr);
  // UnitId id = u->GetId();
  // bool success = (
  //     receiver->CleanupDurativeCmd(env, id) &&
  //     receiver->FinishDurativeCmd(id));
  // return success;
  return true;
}

bool writeGather(
    const GameEnv& env,
    const Unit* u,
    PlayerId playerId,
    UnitId targetId,
    std::map<UnitId, CmdBPtr>* assignedCmds,
    bool verbose) {
  assert(u != nullptr);

  if (u->GetUnitType() != PEASANT) {
    if (verbose) {
      std::cout << "Warning: gather discarded due to invalid unit_type: "
                << u->GetUnitType() << std::endl;
    }
    return false;
  }

  const Unit* target = env.GetUnit(targetId);
  if (target == nullptr) {
    std::cout << "Error: gather non-existing target: " << targetId << std::endl;
    assert(false);
  }
  if (target->GetUnitType() != RESOURCE) {
    std::cout << "Error: gather target is not resource: "
              << target->GetUnitType() << std::endl;
    assert(false);
  }

  // decide town hall
  const Unit* town_hall = env.FindClosestTownHallP(playerId, u->GetPointF());
  if (town_hall == nullptr) {
    if (verbose) {
      std::cout << "Warning: gather, no town_hall found " << std::endl;
    }
    return false;
  }

  (*assignedCmds)[u->GetId()] = _G(town_hall->GetId(), targetId, 1);
  return true;
}

bool writeAttack(
    const GameEnv& env,
    const Unit* u,
    UnitId targetId,
    std::map<UnitId, CmdBPtr>* assignedCmds,
    bool verbose) {
  assert(u != nullptr);

  const auto& gamedef = env.GetGameDef();
  UnitType attackerType = u->GetUnitType();
  if (gamedef.IsUnitTypeBuilding(attackerType) && attackerType != GUARD_TOWER) {
    if (verbose) {
      std::cout << "Warning: attack issued for invalid type: "
                << attackerType << std::endl;
    }
    return false;
  }

  const Unit* target = env.GetUnit(targetId);
  if (target == nullptr) {
    std::cout << "Error: attack target not exist: " << targetId << std::endl;
    assert(false);
  }

  UnitType targetType = u->GetUnitType();
  if (!gamedef.CanAttack(attackerType, targetType)) {
    if (verbose) {
      std::cout << "Warning: " << attackerType << " cannot attack "
                << targetType << std::endl;
    }
    return false;
  }

  (*assignedCmds)[u->GetId()] = _A(targetId);
  return true;
}

bool writeBuildUnit(
    const GameEnv& env,
    const CmdReceiver& receiver,
    Preload* preload,
    const Unit* u,
    UnitType targetType,
    std::map<UnitId, CmdBPtr>* assignedCmds,
    bool verbose) {
  assert(u != nullptr);

  const auto& gamedef = env.GetGameDef();
  UnitType builderType = u->GetUnitType();

  if (!gamedef.IsUnitTypeBuilding(builderType)) {
    if (verbose) {
      std::cout << "Warning: unit builder should be building, got: "
                << builderType << std::endl;
    }
    return false;
  }

  if (targetType == RESOURCE || gamedef.unit(targetType).GetBuildFrom() != builderType) {
    if (verbose) {
      std::cout << "Warning: " << builderType
                << " cannot build " << targetType << std::endl;
    }
    return false;
  }

  bool inBuild = false;
  if (InBuildingUnitType(receiver, *u, targetType, &inBuild)) {
    // building is building the same type of unit
    return true;
  }
  if (inBuild) {
    if (verbose) {
      std::cout << "Warning: in build unit, building is busy: "
                << builderType << std::endl;
    }
    return false;
  }

  if (!preload->Affordable(targetType)) {
    if (verbose) {
      std::cout << "Warning: in build unit, not affordable" << std::endl;
    }
    return false;
  }
  assert(preload->BuildIfAffordable(targetType));

  (*assignedCmds)[u->GetId()] = _B(targetType);
  return true;
}

bool writeBuildBuilding(
    const GameEnv& env,
    const CmdReceiver& receiver,
    Preload* preload,
    const Unit* u,
    UnitType targetType,
    int x,
    int y,
    std::map<UnitId, CmdBPtr>* assignedCmds,
    bool verbose) {
  assert(u != nullptr);

  UnitType builderType = u->GetUnitType();
  if (builderType != PEASANT) {
    if (verbose) {
      std::cout << "Warning: building builder should be peasant, got "
                << builderType << std::endl;
    }
    return false;
  }

  const auto& gamedef = env.GetGameDef();
  if (gamedef.unit(targetType).GetBuildFrom() != builderType) {
    if (verbose) {
      std::cout << "Warning: " << builderType
                << " cannot build " << targetType << std::endl;
    }
    return false;
  }

  if (!env.GetMap().IsIn(x, y)) {
    std::cout << "Error: build loc not in map, got: "
              << x << ", " << y << std::endl;
    assert(false);
  }

  bool inBuild = false;
  if (InBuildingUnitType(receiver, *u, targetType, &inBuild)) {
    // peasant is building the same type of unit
    return true;
  }
  if (inBuild) {
    if (verbose) {
      std::cout << "Warning: in build building, peasant is busy" << std::endl;
    }
    return false;
  }

  if (!preload->Affordable(targetType)) {
    if (verbose) {
      std::cout << "Warning: in build building, target not affordable, resource: "
                << preload->Resource() << std::endl;
    }
    return false;
  }
  assert(preload->BuildIfAffordable(targetType));

  (*assignedCmds)[u->GetId()] = _B(targetType, PointF(x, y));
  return true;
}

bool writeMove(
    const GameEnv& env,
    const Unit* u,
    int x,
    int y,
    std::map<UnitId, CmdBPtr>* assignedCmds,
    bool verbose) {
  assert(u != nullptr);

  UnitType utype = u->GetUnitType();
  if (GameDef::IsUnitTypeBuilding(utype)) {
    if (verbose) {
      std::cout << "Warning: " << utype << " cannit move" << std::endl;
    }
    return false;
  }

  if (!env.GetMap().IsIn(x, y)) {
    std::cout << "Error: dest not in map " << x << ", " << y << std::endl;
    assert(false);
  }

  (*assignedCmds)[u->GetId()] = _M(PointF(x, y));
  return true;
}

void AggregatedCmdReply::logCmds(
    const Preload& preload, const GameEnv& env) const {
  // if (globContinue) {
  //   std::cout << "Global Continue" << std::endl;
  //   return;
  // }

  const auto& myidx2id = preload.getMyidx2id();
  const auto& resourceidx2id = preload.getResourceidx2id();
  const auto& enemyidx2id = preload.getEnemyidx2id();

  for (int i = 0; i < numUnit(); ++i) {
    UnitId unitId = myidx2id.at(i);
    const Unit* u = env.GetUnit(unitId);
    if (u == nullptr) {
      std::cout << "Error: cannot find unit: " << unitId << std::endl;
      assert(false);
    }

    CmdTargetType cmdType = static_cast<CmdTargetType>(cmdType_[i]);
    std::cout << "Unit: " << unitId << ", type: " << u->GetUnitType()
              << ", cmd: " << cmdType << ", ";

    if (cmdType == CMD_TARGET_IDLE || cmdType == CMD_TARGET_CONT) {
      std::cout << std::endl;
      continue;
    }

    if (cmdType == CMD_TARGET_GATHER) {
      if (targetIdx_[i] >= (int)resourceidx2id.size()) {
        if (resourceidx2id.empty()) {
          std::cout << "Warning: no resource to gather" << std::endl;
        } else {
          std::cout << "Error: gather invalid idx" << std::endl;
          assert(false);
        }
        continue;
      }

      UnitId targetId = resourceidx2id[targetIdx_[i]];
      const Unit* target = env.GetUnit(targetId);
      if (target == nullptr) {
        std::cout << "Error: gather non-existing target: " << targetId << std::endl;
        assert(false);
      }
      if (target->GetUnitType() != RESOURCE) {
        std::cout << "Error: gather target is not resource: "
                  << target->GetUnitType() << std::endl;
        assert(false);
      }
      std::cout << "RESOURCE at x=" << target->GetPointF().x
                << ", y=" << target->GetPointF().y << std::endl;
    } else if (cmdType == CMD_TARGET_ATTACK) {
      if (targetIdx_[i] >= (int)enemyidx2id.size()) {
        if (enemyidx2id.empty()) {
          std::cout << "Warning: no enemy to attack" << std::endl;
        } else {
          std::cout << "Error: attack invalid idx" << std::endl;
          assert(false);
        }
        continue;
      }

      UnitId targetId = enemyidx2id[targetIdx_[i]];
      const Unit* target = env.GetUnit(targetId);
      if (target == nullptr) {
        std::cout << "Error: attack non-existing target: " << targetId << std::endl;
        assert(false);
      }
      std::cout << target->GetUnitType() << std::endl;
    } else if (cmdType == CMD_TARGET_BUILD_BUILDING) {
      UnitType unitType = static_cast<UnitType>(targetType_[i]);
      std::cout << "type: " << unitType
                << " dest: x=" << targetX_[i]
                << ", y=" << targetY_[i] << std::endl;
    } else if (cmdType == CMD_TARGET_BUILD_UNIT) {
      UnitType unitType = static_cast<UnitType>(targetType_[i]);
      std::cout << "type: " << unitType << std::endl;
    } else if (cmdType == CMD_TARGET_MOVE) {
      std::cout << "dest: x=" << targetX_[i] << ", y=" << targetY_[i] << std::endl;
    } else {
      std::cout << "Invalid cmd type: " << cmdType << std::endl;
      assert(false);
    }
  }
}

bool AggregatedCmdReply::sampleCmds(std::mt19937*) {
  auto contSample = contProb_->getBuffer().multinomial(1);
  bool globCont = contSample.item<int64_t>() == 1;

  if (verbose) {
    std::cout << "Glob Continue Prob:\n" << contProb_->getBuffer() << std::endl;
    std::cout << "Glob Continue: " << std::boolalpha << globCont << std::endl;
  }

  if (globCont || numUnit() == 0) {
    // same as idle, nothing to do here
    if (numUnit() == 0) {
      std::cout << ">>>Warning:: no units in sampleCmds" << std::endl;
    }
    return true;
  }

  auto cmdTypeSample = cmdTypeProb_->getBuffer().multinomial(1);
  bool allCont = true;
  for (int i = 0; i < numUnit(); ++i) {
    // std::cout << "cmd type prob:" << cmdTypeProb_->getBuffer()[i] << std::endl;
    auto cmdType = static_cast<CmdTargetType>(
        cmdTypeSample[i][0].template item<int64_t>());
    cmdType_[i] = cmdType;

    if (cmdType != CMD_TARGET_CONT) {
      allCont = false;
    }

    if (cmdType == CMD_TARGET_IDLE || cmdType == CMD_TARGET_CONT) {
      // jump to next directly
      continue;
    } else if (cmdType == CMD_TARGET_GATHER) {
      targetIdx_[i] = gatherIdxProb_->getBuffer()[i].multinomial(1).template item<int64_t>();
    } else if (cmdType == CMD_TARGET_ATTACK) {
      targetIdx_[i] = attackIdxProb_->getBuffer()[i].multinomial(1).template item<int64_t>();
    } else if (cmdType == CMD_TARGET_BUILD_UNIT) {
      targetType_[i] = unitTypeProb_->getBuffer()[i].multinomial(1).template item<int64_t>();
    } else if (cmdType == CMD_TARGET_BUILD_BUILDING) {
      targetType_[i] = buildingTypeProb_->getBuffer()[i].multinomial(1).template item<int64_t>();
      int64_t loc = buildingLocProb_->getBuffer()[i].multinomial(1).template item<int64_t>();
      targetY_[i] = loc / mapX;
      targetX_[i] = loc % mapX;
    } else if (cmdType == CMD_TARGET_MOVE) {
      // need to sample locations
      int64_t loc = moveLocProb_->getBuffer()[i].multinomial(1).template item<int64_t>();
      targetY_[i] = loc / mapX;
      targetX_[i] = loc % mapX;
    }
  }
  return !allCont;
}

void AggregatedCmdReply::writeCmds(
    const GameEnv& env,
    const CmdReceiver& receiver,
    PlayerId playerId,
    Preload* preload,
    std::map<UnitId, CmdBPtr>* assignedCmds,
    std::mt19937* rng) {
  assert(numUnit() >= 0 && numUnit() <= maxNumUnit);

  // if (useProb_) {
  for (int trial = 0; trial < 10; ++trial) {
    bool success = sampleCmds(rng);
    if (!success) {
      if (verbose) {
        std::cout << "--sample rejected--" << std::endl;
      }
      clearCmds();
    } else {
      break;
    }
  }

  if (verbose) {
    logCmds(*preload, env);
  }

  const auto& myidx2id = preload->getMyidx2id();
  const auto& enemyidx2id = preload->getEnemyidx2id();
  const auto& resourceidx2id = preload->getResourceidx2id();

  for (int i = 0; i < numUnit(); ++i) {
    // cmd for unit should never be INVALID
    if (cmdType_[i] < 0 || cmdType_[i] >= (int)NUM_CMD_TARGET_TYPE) {
      std::cout << "Error: cmdType out of range: " << cmdType_[i] << std::endl;
      assert(false);
    }

    CmdTargetType cmdType = static_cast<CmdTargetType>(cmdType_[i]);
    // UnitId unitId = static_cast<UnitId>(unitId_[i]);
    UnitId unitId = myidx2id.at(i);
    const Unit* u = env.GetUnit(unitId);
    if (u == nullptr) {
      std::cout << "Error: cannot find unit: " << unitId << std::endl;
      assert(false);
    }
    if (u->GetPlayerId() != playerId) {
      std::cout << "Error: cannot control enemy's unit: " << unitId << std::endl;
      assert(false);
    }

    if (UnitHasCmd(unitId, *assignedCmds)) {
      std::cout << "Error: unit: " << unitId << " has pending cmd" << std::endl;
      assert(false);
    }

    if (cmdType == CMD_TARGET_IDLE || cmdType == CMD_TARGET_CONT) {
      writeIdle(env, u, receiver, verbose);
    } else if (cmdType == CMD_TARGET_GATHER) {
      if (targetIdx_[i] >= (int)resourceidx2id.size()) {
        if (resourceidx2id.empty()) {
          if (verbose) {
            std::cout << "Warning: no resource to gather" << std::endl;
          }
        } else {
          std::cout << "Error: gather invalid idx" << std::endl;
          assert(false);
        }
        continue;
      }
      UnitId targetId = resourceidx2id[targetIdx_[i]];
      writeGather(env, u, playerId, targetId, assignedCmds, verbose);
    } else if (cmdType == CMD_TARGET_ATTACK) {
      if (targetIdx_[i] >= (int)enemyidx2id.size()) {
        if (enemyidx2id.empty()) {
          if (verbose) {
            std::cout << "Warning: no enemy to attack" << std::endl;
          }
        } else {
          std::cout << "Error: attack invalid idx" << std::endl;
          assert(false);
        }
        continue;
      }
      UnitId targetId = enemyidx2id[targetIdx_[i]];
      writeAttack(env, u, targetId, assignedCmds, verbose);
    } else if (cmdType == CMD_TARGET_BUILD_UNIT || cmdType == CMD_TARGET_BUILD_BUILDING) {
      assert(targetType_[i] >= 0 && targetType_[i] < (int)NUM_MINIRTS_UNITTYPE);
      UnitType targetType = static_cast<UnitType>(targetType_[i]);
      if (targetType == AVIARY || targetType == ARCHERY) {
        if (verbose) {
          std::cout << "Warning: invalid type for build: " << targetType << std::endl;
        }
        continue;
      }

      if (cmdType == CMD_TARGET_BUILD_UNIT) {
        writeBuildUnit(env, receiver, preload, u, targetType, assignedCmds, verbose);
      } else if (cmdType == CMD_TARGET_BUILD_BUILDING) {
        int x = targetX_[i];
        int y = targetY_[i];
        writeBuildBuilding(
            env, receiver, preload, u, targetType, x, y, assignedCmds, verbose);
      }
    } else if (cmdType == CMD_TARGET_MOVE) {
      writeMove(env, u, targetX_[i], targetY_[i], assignedCmds, verbose);
    } else {
      std::cout << "Error: wrong cmd type: " << cmdType << std::endl;
      assert(false);
    }
  }
}
