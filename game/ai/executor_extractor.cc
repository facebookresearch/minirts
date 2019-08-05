// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#include "ai/executor_extractor.h"
#include "engine/cmd_target.h"

const int ExecutorExtractor::mapCoordOffset = 0;
const int ExecutorExtractor::mapVisibilityOffset = 2;
const int ExecutorExtractor::mapTerrainOffset = mapVisibilityOffset + NUM_VISIBILITY;
const int ExecutorExtractor::mapArmyOffset = mapTerrainOffset + NUM_TERRAIN;
const int ExecutorExtractor::mapEnemyOffset = mapArmyOffset + (NUM_MINIRTS_UNITTYPE - 1);
const int ExecutorExtractor::mapResourceOffset = mapEnemyOffset + (NUM_MINIRTS_UNITTYPE - 1);
const int ExecutorExtractor::mapFeatNumChannels = mapResourceOffset + 1;

const int ExecutorExtractor::countVisibilityOffset = 0;
const int ExecutorExtractor::countArmyOffset = countVisibilityOffset + NUM_VISIBILITY;
const int ExecutorExtractor::countEnemyOffset = countArmyOffset + (NUM_MINIRTS_UNITTYPE - 1);
const int ExecutorExtractor::countResourceOffset = countEnemyOffset + (NUM_MINIRTS_UNITTYPE - 1);
const int ExecutorExtractor::countFeatNumChannels = countResourceOffset + 1;

ExecutorExtractor::ExecutorExtractor(
    int maxNumUnits,
    int numPrevCmds,
    int mapX,
    int mapY,
    int numResourceBins,
    int resourceBinSize,
    int numInstructions,
    int maxRawChars,
    int tLen,
    int numCmdTypes,
    int numUnitTypes,
    bool useMovingAvg,
    float movingAvgDecay,
    bool verbose,
    std::string prefix)
    : maxNumUnits_(maxNumUnits),
      numPrevCmds_(numPrevCmds),
      mapX_(mapX),
      mapY_(mapY),
      numResourceBins_(numResourceBins),
      resourceBinSize_(resourceBinSize),
      useMovingAvg_(useMovingAvg),
      movingAvgDecay_(movingAvgDecay),
      verbose_(verbose),
      // coach policy and executor policy
      instruction_(numInstructions, -1, tLen, maxRawChars),
      cmdReply_(maxNumUnits, numCmdTypes, numUnitTypes, mapX, mapY, verbose),
      // count features
      countFeat_(prefix+"count", tLen + 1, {countFeatNumChannels}, torch::kFloat32),
      baseCountFeat_(prefix+"base_count", tLen + 1, {countFeatNumChannels}, torch::kFloat32),
      consCount_(prefix+"cons_count", tLen + 1, {numUnitTypes}, torch::kFloat32),
      movingEnemyCount_(prefix+"moving_enemy_count", tLen + 1, {numUnitTypes}, torch::kFloat32),
      // executor & coach features
      mapFeat_(prefix+"map", tLen + 1, {mapFeatNumChannels, mapY, mapX}, torch::kFloat32),
      // army basic
      numArmy_(prefix+"num_army", tLen + 1, {1}, torch::kInt64),
      armyType_(prefix+"army_type", tLen + 1, {maxNumUnits}, torch::kInt64),
      armyHp_(prefix+"army_hp", tLen + 1, {maxNumUnits}, torch::kFloat32),
      armyX_(prefix+"army_x", tLen + 1, {maxNumUnits}, torch::kInt64),
      armyY_(prefix+"army_y", tLen + 1, {maxNumUnits}, torch::kInt64),
      // current cmd & prev cmd
      cCmdType_(prefix+"current_cmd_type", tLen + 1, {maxNumUnits}, torch::kInt64),
      cCmdUnitType_(prefix+"current_cmd_unit_type", tLen + 1, {maxNumUnits}, torch::kInt64),
      cCmdX_(prefix+"current_cmd_x", tLen + 1, {maxNumUnits}, torch::kInt64),
      cCmdY_(prefix+"current_cmd_y", tLen + 1, {maxNumUnits}, torch::kInt64),
      cCmdGatherIdx_(prefix+"current_cmd_gather_idx", tLen + 1, {maxNumUnits}, torch::kInt64),
      cCmdAttackIdx_(prefix+"current_cmd_attack_idx", tLen + 1, {maxNumUnits}, torch::kInt64),
      pCmdType_(prefix+"prev_cmd", tLen + 1, {maxNumUnits, numPrevCmds}, torch::kInt64),
      // enemy basic
      numEnemy_(prefix+"num_enemy", tLen + 1, {1}, torch::kInt64),
      enemyType_(prefix+"enemy_type", tLen + 1, {maxNumUnits}, torch::kInt64),
      enemyHp_(prefix+"enemy_hp", tLen + 1, {maxNumUnits}, torch::kFloat32),
      enemyX_(prefix+"enemy_x", tLen + 1, {maxNumUnits}, torch::kInt64),
      enemyY_(prefix+"enemy_y", tLen + 1, {maxNumUnits}, torch::kInt64),
      // resource unit
      numResource_(prefix+"num_resource", tLen + 1, {1}, torch::kInt64),
      resourceType_(prefix+"resource_type", tLen + 1, {maxNumUnits}, torch::kInt64),
      resourceHp_(prefix+"resource_hp", tLen + 1, {maxNumUnits}, torch::kFloat32),
      resourceX_(prefix+"resource_x", tLen + 1, {maxNumUnits}, torch::kInt64),
      resourceY_(prefix+"resource_y", tLen + 1, {maxNumUnits}, torch::kInt64),
      // resource bin
      resourceBin_(prefix+"resource_bin", tLen + 1, {numResourceBins}, torch::kFloat32),
      // reward & terminal
      reward_(prefix+"reward", tLen, {1}, torch::kFloat32),
      terminal_(prefix+"terminal", tLen, {1}, torch::kFloat32)
{
  // TODO: dirty hack for safety check
  assert(numPrevCmds_ == 25);
  // if (movingAvgDecay_ != 0.98) {
  //   std::cout << "moving avg decay: " << movingAvgDecay_ << std::endl;
  // }
  assert(std::abs(movingAvgDecay_ - 0.98) < 1e-6);

  features_.push_back(std::ref(countFeat_));
  features_.push_back(std::ref(baseCountFeat_));
  features_.push_back(std::ref(consCount_));
  features_.push_back(std::ref(movingEnemyCount_));

  features_.push_back(std::ref(mapFeat_));

  features_.push_back(std::ref(numArmy_));
  features_.push_back(std::ref(armyType_));
  features_.push_back(std::ref(armyHp_));
  features_.push_back(std::ref(armyX_));
  features_.push_back(std::ref(armyY_));

  features_.push_back(std::ref(cCmdType_));
  features_.push_back(std::ref(cCmdUnitType_));
  features_.push_back(std::ref(cCmdX_));
  features_.push_back(std::ref(cCmdY_));
  features_.push_back(std::ref(cCmdGatherIdx_));
  features_.push_back(std::ref(cCmdAttackIdx_));
  features_.push_back(std::ref(pCmdType_));

  features_.push_back(std::ref(numEnemy_));
  features_.push_back(std::ref(enemyType_));
  features_.push_back(std::ref(enemyHp_));
  features_.push_back(std::ref(enemyX_));
  features_.push_back(std::ref(enemyY_));

  features_.push_back(std::ref(numResource_));
  features_.push_back(std::ref(resourceType_));
  features_.push_back(std::ref(resourceHp_));
  features_.push_back(std::ref(resourceX_));
  features_.push_back(std::ref(resourceY_));

  features_.push_back(std::ref(resourceBin_));
}

bool ExecutorExtractor::skim(const RTSStateExtend& state, const AI& ai, bool nofow) {
  if (!useMovingAvg_) {
    return false;
  }

  bool fow = ((!nofow) && ai.respectFow());
  Preload preload;
  preload.GatherInfo(
      state.env(),
      ai.getId(),
      state.receiver(),
      ai.buildQueue,
      fow);

  const auto& enemyTroops = preload.EnemyTroops();
  auto movingEnemyCount = movingEnemyCount_.getBuffer().accessor<float, 1>();
  assert(movingEnemyCount.size(0) == (int)enemyTroops.size());
  for (int i = 0; i < (int)enemyTroops.size(); ++i) {
    int count = enemyTroops[i].size();
    // if (count != 0) {
    //   std::cout << "before: " << movingEnemyCount[i] << std::endl;
    // }
    movingEnemyCount[i] *= movingAvgDecay_;
    movingEnemyCount[i] += count * (1 - movingAvgDecay_);
    // if (count != 0) {
    //   std::cout << "after: " << movingEnemyCount[i] << std::endl;
    // }
  }
  // std::cout << "moving enemy c++ " << std::endl;
  // std::cout << movingEnemyCount_.getBuffer() << std::endl;
  return true;
}

void ExecutorExtractor::reset() {
  clearFeatures();
  cmdReply_.reset();
}

void ExecutorExtractor::newGame() {
  // std::cout << "=================NEW GAME===============" << std::endl;
  // reset volatile feature / buffer
  reset();

  // clear persistent feature / buffer
  baseCountFeat_.getBuffer().zero_();
  movingEnemyCount_.getBuffer().zero_();
  prevCmds_.clear();

  instruction_.newGame();

  gameStart_ = true;
  townHallDiscoveredTick_ = -1;
  prevExploredMap_ = 0.0;
  exploredMap_ = 0.0;
}

// append current feature to trajectory
void ExecutorExtractor::pushGameFeature() {
  int idx = -1;
  for (auto& feat : features_) {
    int fidx = feat.get().pushBufferToTrajectory();
    if (idx == -1) {
      idx = fidx;
    } else {
      assert(idx == fidx);
    }
  }

  instruction_.pushGameFeature();
}

void ExecutorExtractor::pushLastRAndTerminal() {
  int idx = terminal_.pushBufferToTrajectory();
  assert(idx == reward_.pushBufferToTrajectory());
}

void ExecutorExtractor::pushActionAndPolicy() {
  instruction_.pushActionAndPolicy();
}

void ExecutorExtractor::postActUpdate() {
  instruction_.postActUpdate();
  if (!instruction_.sameInstruction()) {
    baseCountFeat_.getBuffer().copy_(countFeat_.getBuffer());
    // clearPrevCmd();
  }
  prevExploredMap_ = exploredMap_;
}

void ExecutorExtractor::updatePrevCmd(
    const std::map<UnitId, CmdBPtr>& assignedCmds) {
  for (auto& id2pcmd : prevCmds_) {
    UnitId id = id2pcmd.first;
    // std::cout << "finding for " << id << std::endl;
    auto finder = assignedCmds.find(id);
    if (finder == assignedCmds.end()) {
      // std::cout << "found no new cmd" << std::endl;
      id2pcmd.second.push_back(CMD_TARGET_CONT);
    } else {
      CmdTargetType cmdType = getCmdTargetType(finder->second);
      // std::cout << "found " << cmdType << std::endl;
      id2pcmd.second.push_back(cmdType);
    }
  }

  // fill in cmds for new units
  // this should never happen
  for (const auto& id2cmd : assignedCmds) {
    UnitId id = id2cmd.first;
    // CmdTargetType cmdType = getCmdTargetType(id2cmd.second);
    // std::cout << "cmd type to add: " << cmdType << std::endl;
    auto finder = prevCmds_.find(id);
    if (finder == prevCmds_.end()) {
      std::cout << "Error: new unit found in assigned Cmds" << std::endl;
    }
    assert(finder != prevCmds_.end());
  }
}

// (TODO: for now keep using setUnitId2IdxMaps for sanity check)
// UnitId2Idx maps will be set in this function,
// no need to call preload.setUnitId2IdxMaps()
void ExecutorExtractor::computeFeatures(
    Preload& preload,
    const CmdReceiver& receiver,
    const GameEnv& env,
    PlayerId playerId,
    bool respectFow) {
  preload.setUnitId2IdxMaps();

  computeArmyAndResourceBasic(preload);

  computeArmyExtra(preload, receiver, env, playerId, respectFow);

  computeEnemyBasic(receiver.GetTick(), preload);

  computeMapFeature(preload, env, playerId);

  computeResourceBin(preload);

  computeLastRAndTerminal(receiver.GetTick(), env, playerId);

  // TODO: seperate out unit_count feat
  computeUnitConsCount(preload);

  if (gameStart_) {
    baseCountFeat_.getBuffer().copy_(countFeat_.getBuffer());
    gameStart_ = false;
  }
}

void ExecutorExtractor::computeLastRAndTerminal(int, const GameEnv& env, PlayerId playerId) {
  if (env.GetTermination()) {
    terminal_.getBuffer()[0] = (float)true;
    auto winner = env.GetWinnerId();
    reward_.getBuffer()[0] = (winner == playerId ? 1.0 : -1.0);
  } else {
    terminal_.getBuffer()[0] = (float)false;
    reward_.getBuffer()[0] = 0;
  }
  // if (tick == townHallDiscoveredTick_) {
  //   // std::cout << "<<<<<<<" << tick << ", " << townHallDiscoveredTick_ << std::endl;
  //   reward_.getBuffer()[0] += 500 * 1.0 / townHallDiscoveredTick_;
  // }
  // reward_.getBuffer()[0] += 0.1 * (exploredMap_ - prevExploredMap_);
  // std::cout << "reward: " << exploredMap_ - prevExploredMap_ << std::endl;
}

void ExecutorExtractor::computeArmyAndResourceBasic(const Preload& preload) {
  const auto& id2idx = preload.getUnitId2Idx();
  const auto& myTroops = preload.MyTroops();

  auto countFeat = countFeat_.getBuffer().accessor<float, 1>();

  auto resourceType = resourceType_.getBuffer().accessor<int64_t, 1>();
  auto resourceHp = resourceHp_.getBuffer().accessor<float, 1>();
  auto resourceX = resourceX_.getBuffer().accessor<int64_t, 1>();
  auto resourceY = resourceY_.getBuffer().accessor<int64_t, 1>();

  auto armyType = armyType_.getBuffer().accessor<int64_t, 1>();
  auto armyHp = armyHp_.getBuffer().accessor<float, 1>();
  auto armyX = armyX_.getBuffer().accessor<int64_t, 1>();
  auto armyY = armyY_.getBuffer().accessor<int64_t, 1>();

  int numResource = 0;
  int numArmy = 0;
  for (int typeIdx = 0; typeIdx < (int)myTroops.size(); ++typeIdx) {
    auto utype = static_cast<UnitType>(typeIdx);

    // count feature
    {
      int offset = 0;
      if (utype == RESOURCE) {
        assert(typeIdx == 0);
        offset = countResourceOffset;
      } else {
        offset = countArmyOffset + typeIdx - 1;
      }
      countFeat[offset] = myTroops[typeIdx].size();
    }

    for (const Unit* u : myTroops[typeIdx]) {
      assert(utype == u->GetUnitType());
      const UnitId id = u->GetId();
      const int idx = id2idx.at(id);
      float hp = u->GetNormalizedHp();
      auto p = u->GetCorrectedPointF();

      if (utype == RESOURCE) {
        // assert(idx == (int)resourceType_.size());
        ++numResource;
        assert(idx < resourceType.size(0));
        resourceType[idx] = utype;
        resourceHp[idx] = hp;
        resourceX[idx] = p.x;
        resourceY[idx] = p.y;
      } else {
        // part of my army
        // assert(idx == (int)armyType_.size());
        ++numArmy;
        assert(idx < armyType.size(0));
        armyType[idx] = utype;
        armyHp[idx] = hp;
        armyX[idx] = p.x;
        armyY[idx] = p.y;
      }
    }
  }

  numResource_.getBuffer()[0] = numResource;
  numArmy_.getBuffer()[0] = numArmy;
}

void ExecutorExtractor::computeEnemyBasic(int tick, const Preload& preload) {
  // enemy_unit
  const auto& id2idx = preload.getUnitId2Idx();
  const auto& enemyTroops = preload.EnemyTroops();

  auto countFeat = countFeat_.getBuffer().accessor<float, 1>();

  auto enemyType = enemyType_.getBuffer().accessor<int64_t, 1>();
  auto enemyHp = enemyHp_.getBuffer().accessor<float, 1>();
  auto enemyX = enemyX_.getBuffer().accessor<int64_t, 1>();
  auto enemyY = enemyY_.getBuffer().accessor<int64_t, 1>();

  int numEnemy = 0;
  for (int typeIdx = 0; typeIdx < (int)enemyTroops.size(); ++typeIdx) {
    {
      UnitType utype = static_cast<UnitType>(typeIdx);
      if (townHallDiscoveredTick_ == -1
          && (utype == TOWN_HALL
              || utype == BARRACK
              || utype == STABLE
              || utype == BLACKSMITH
              || utype == WORKSHOP)
          && enemyTroops[typeIdx].size() >= 1) {
        townHallDiscoveredTick_ = tick;
      }

      if (utype != RESOURCE) {
        int offset = countEnemyOffset + typeIdx - 1;
        countFeat[offset] = enemyTroops[typeIdx].size();
      } else {
        assert(enemyTroops[typeIdx].size() == 0);
        continue;
      }
    }

    for (const auto& u : enemyTroops[typeIdx]) {
      ++numEnemy;
      const UnitId id = u->GetId();
      const int idx = id2idx.at(id);
      UnitType utype = u->GetUnitType();
      float hp = u->GetNormalizedHp();
      auto p = u->GetCorrectedPointF();

      // assert(idx == (int)enemyType_.size());
      assert(idx < enemyType.size(0));

      enemyType[idx] = utype;
      enemyHp[idx] = hp;
      enemyX[idx] = p.x;
      enemyY[idx] = p.y;
    }
  }
  numEnemy_.getBuffer()[0] = numEnemy;
}

void ExecutorExtractor::computeArmyExtra(
    const Preload& preload,
    const CmdReceiver& receiver,
    const GameEnv& env,
    PlayerId playerId,
    bool respectFow) {
  GameEnvAspect aspect(env, playerId, respectFow);
  const auto& id2idx = preload.getUnitId2Idx();

  auto cCmdType = cCmdType_.getBuffer().accessor<int64_t, 1>();
  auto cCmdUnitType = cCmdUnitType_.getBuffer().accessor<int64_t, 1>();
  auto cCmdX = cCmdX_.getBuffer().accessor<int64_t, 1>();
  auto cCmdY = cCmdY_.getBuffer().accessor<int64_t, 1>();
  auto cCmdGatherIdx = cCmdGatherIdx_.getBuffer().accessor<int64_t, 1>();
  auto cCmdAttackIdx = cCmdAttackIdx_.getBuffer().accessor<int64_t, 1>();

  auto pCmdType = pCmdType_.getBuffer().accessor<int64_t, 2>();
  const auto& myTroops = preload.MyTroops();
  for (int typeIdx = 0; typeIdx < (int)myTroops.size(); ++typeIdx) {
    auto utype = static_cast<UnitType>(typeIdx);
    if (utype == RESOURCE) {
      continue;
    }

    for (const Unit* u : myTroops[typeIdx]) {
      const UnitId id = u->GetId();
      const int idx = id2idx.at(id);

      // get current cmd and write to feature
      CmdTarget currentCmd = CreateCmdTargetForUnit(aspect, receiver, *u);
      assert(currentCmd.unitId == id);
      assert(currentCmd.cmdType != CMD_TARGET_CONT);

      cCmdType[idx] = (int64_t)currentCmd.cmdType;
      cCmdUnitType[idx] = (int64_t)currentCmd.targetType;
      cCmdX[idx] = (int64_t)currentCmd.targetX;
      cCmdY[idx] = (int64_t)currentCmd.targetY;

      if (currentCmd.cmdType == CMD_TARGET_GATHER) {
        cCmdGatherIdx[idx] = id2idx.at(currentCmd.targetId);
      }
      if (currentCmd.cmdType == CMD_TARGET_ATTACK) {
        cCmdAttackIdx[idx] = id2idx.at(currentCmd.targetId);
      }

      // modify prevCmd_
      auto prevCmdFinder = prevCmds_.find(id);
      if (prevCmdFinder == prevCmds_.end()) {
        // this is a new unit
        prevCmds_[id] = std::vector<int64_t>();
        prevCmds_[id].push_back((int64_t)currentCmd.cmdType);
      }
      // prevCmds_[id].push_back((int64_t)currentCmd.cmdType);
      const auto& unitPrevCmd = prevCmds_[id];

      // fill in pCmdType feature
      int srcOffset = 0;
      int destOffset = 0;
      int numCmd = unitPrevCmd.size();
      if (numCmd > numPrevCmds_) {
        srcOffset = numCmd - numPrevCmds_;
        numCmd = numPrevCmds_;
      } else {
        destOffset = numPrevCmds_ - numCmd;
      }
      for (int i = 0; i < numCmd; ++i) {
        assert(i + srcOffset < (int)unitPrevCmd.size());
        assert(i + destOffset < numPrevCmds_);
        pCmdType[idx][i + destOffset] = unitPrevCmd[i + srcOffset];
      }
    }
  }
}

void ExecutorExtractor::computeUnitConsCount(const Preload& preload) {
  const std::vector<int>& counters = preload.CntUnderConstruction();
  auto consCount = consCount_.getBuffer().accessor<float, 1>();

  for (int i = 0; i < (int)counters.size(); ++i) {
    consCount[i] = (float)counters[i];
  }
}

void ExecutorExtractor::computeResourceBin(const Preload& preload) {
  // resource
  float resource = preload.Resource();
  int binIdx = resource / resourceBinSize_;
  binIdx = std::min(binIdx, numResourceBins_ - 1);

  auto accessor = resourceBin_.getBuffer().accessor<float, 1>();
  assert(numResourceBins_ == accessor.size(0));
  for (int i = 0; i < numResourceBins_; ++i) {
    if (binIdx == i) {
      accessor[i] = 1;
    } else {
      accessor[i] = 0;
    }
  }
}

void ExecutorExtractor::computeMapFeature(
    const Preload& preload, const GameEnv& env, PlayerId playerId) {
  const Player& player  = env.GetPlayer(playerId);
  const RTSMap& map = player.GetMap();
  assert(map.GetYSize() == mapY_);
  assert(map.GetXSize() == mapX_);

  auto mapFeat = mapFeat_.getBuffer().accessor<float, 3>();
  auto countFeat = countFeat_.getBuffer().accessor<float, 1>();
  assert((int)mapFeat.size(0) == mapFeatNumChannels);
  assert((int)mapFeat.size(1) == mapY_);
  assert((int)mapFeat.size(2) == mapX_);

  for (int y = 0; y < mapY_; ++y) {
    for (int x = 0; x < mapX_; ++x) {
      mapFeat[0][y][x] = (float)y / mapY_;
      mapFeat[1][y][x] = (float)x / mapX_;
      Loc loc = map.GetLoc(Coord(x, y, 0));
      const Fog& f = player.GetFog(loc);
      Visibility v = f.GetVisibility();
      Terrain t = FOG;
      if (f.HasSeenTerrain()) {
        t = map(loc).type;
      }
      mapFeat[mapVisibilityOffset + (int)v][y][x] = 1;
      mapFeat[mapTerrainOffset + (int)t][y][x] = 1;
      countFeat[countVisibilityOffset + (int)v] += 1;
    }
  }

  {
    int numVis = countFeat[countVisibilityOffset + (int)VISIBLE];
    int numSeen = countFeat[countVisibilityOffset + (int)SEEN];
    int numInvis = countFeat[countVisibilityOffset + (int)INVISIBLE];
    assert(mapY_ * mapX_ == numVis + numSeen + numInvis);
    exploredMap_ = 1 - float(numInvis) / (mapY_ * mapX_);
  }

  countFeat[countVisibilityOffset + (int)VISIBLE] /= (mapY_ * mapX_);
  countFeat[countVisibilityOffset + (int)SEEN] /= (mapY_ * mapX_);
  countFeat[countVisibilityOffset + (int)INVISIBLE] /= (mapY_ * mapX_);

  // fill in units
  const auto& myTroops = preload.MyTroops();
  for (int typeIdx = 0; typeIdx < (int)myTroops.size(); ++typeIdx) {
    UnitType utype = static_cast<UnitType>(typeIdx);
    for (const Unit* u : myTroops[typeIdx]) {
      auto p = u->GetCorrectedPointF();
      int offset = 0;
      if (utype == RESOURCE) {
        assert(typeIdx == 0);
        offset = mapResourceOffset;
      } else {
        offset = mapArmyOffset + typeIdx - 1;  // off by resource
      }
      mapFeat[offset][p.y][p.x] += 1;
    }
  }

  const auto& enemyTroops = preload.EnemyTroops();
  for (int typeIdx = 0; typeIdx < (int)enemyTroops.size(); ++typeIdx) {
    const auto& units = enemyTroops[typeIdx];
    for (const auto& u : units) {
      assert(typeIdx > 0);
      int offset = mapEnemyOffset + typeIdx - 1;  // off by resource
      auto p = u->GetCorrectedPointF();
      mapFeat[offset][p.y][p.x] += 1;
    }
  }
}

// should be called after every act step
void ExecutorExtractor::clearFeatures() {
  countFeat_.getBuffer().zero_();
  // baseCountFeat_.getBuffer().zero_();
  consCount_.getBuffer().zero_();
  // movingEnemyCount_.getBuffer().zero_();

  mapFeat_.getBuffer().zero_();

  numArmy_.getBuffer().zero_();
  armyType_.getBuffer().zero_();
  armyHp_.getBuffer().zero_();
  armyX_.getBuffer().zero_();
  armyY_.getBuffer().zero_();

  cCmdType_.getBuffer().zero_();
  cCmdUnitType_.getBuffer().zero_();
  cCmdX_.getBuffer().zero_();
  cCmdY_.getBuffer().zero_();
  cCmdGatherIdx_.getBuffer().zero_();
  cCmdAttackIdx_.getBuffer().zero_();
  pCmdType_.getBuffer().zero_();

  numEnemy_.getBuffer().zero_();
  enemyType_.getBuffer().zero_();
  enemyHp_.getBuffer().zero_();
  enemyX_.getBuffer().zero_();
  enemyY_.getBuffer().zero_();

  numResource_.getBuffer().zero_();
  resourceType_.getBuffer().zero_();
  resourceHp_.getBuffer().zero_();
  resourceX_.getBuffer().zero_();
  resourceY_.getBuffer().zero_();

  resourceBin_.getBuffer().zero_();
}
