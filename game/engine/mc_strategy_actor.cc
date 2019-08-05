// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#include "mc_strategy_actor.h"

void MCStrategyActor::BuildPeasantIfNeeded(int min_num_peasant,
                                           std::vector<int64_t>* action) {
  int num_peasant = _preload.NumUnit(PEASANT, true);
  if (num_peasant < min_num_peasant && _preload.WithinBudget(PEASANT)) {
    action->at(STATE_BUILD_PEASANT) = 1;
    _preload.DeductBudget(PEASANT);
  }
}

bool MCStrategyActor::DefendIfNeeded(std::vector<int64_t>* action) {
  const auto& enemy_in_range_targets = _preload.EnemyInRangeTargets();
  const auto& enemy_defend_targets = _preload.EnemyDefendTargets();
  bool needDefend = false;

  if (!enemy_defend_targets.empty()) {
    action->at(STATE_DEFEND) = 1;
    needDefend = true;
  }

  if (!enemy_in_range_targets.empty()) {
    action->at(STATE_ATTACK_IN_RANGE) = 1;
    needDefend = true;
  }

  return needDefend;
}

void MCStrategyActor::BuildScoutAttack(const GameDef& gamedef,
                                       UnitType unitType,
                                       std::vector<int64_t>* action,
                                       int minArmy,
                                       int maxArmy) {
  assert(_preload.Ready());
  assert(minArmy > 0);

  auto factoryType = gamedef.GetBuildFrom(unitType);
  int numFactory = _preload.NumUnit(factoryType, true);
  if (numFactory < 1 && _preload.WithinBudget(factoryType)) {
    action->at(StateBuildMap.at(factoryType)) = 1;
    _preload.DeductBudget(factoryType);
  }

  int numBuiltFactory = _preload.NumUnit(factoryType, false);
  const auto& myTroops = _preload.MyTroops();
  const auto& myArmy = _preload.MyArmy();
  if (numBuiltFactory >= 1 && _preload.WithinBudget(unitType) &&
      (maxArmy < 0 || (int)myArmy.size() < maxArmy)) {
    action->at(StateBuildMap.at(unitType)) = 1;
    _preload.DeductBudget(unitType);
  }

  if ((int)myArmy.size() >= minArmy || myTroops[RESOURCE].empty() ||
      myTroops[PEASANT].empty()) {
    if (!needDefend_ && _preload.EnemyTroops()[TOWN_HALL].size() == 0) {
      action->at(STATE_SCOUT) = 1;
    }
    action->at(STATE_ATTACK_BASE) = 1;
  }
}

void MCStrategyActor::BuildScoutAttackTwoUnits(const GameDef& gamedef,
                                               UnitType unitType1,
                                               UnitType unitType2,
                                               std::vector<int64_t>* action,
                                               int minArmy1,
                                               int minArmy2,
                                               int maxArmy) {
  assert(_preload.Ready());
  assert(minArmy1 > 0 && minArmy2 > 0);

  auto factoryType1 = gamedef.GetBuildFrom(unitType1);
  int numFactory1 = _preload.NumUnit(factoryType1, true);
  if (numFactory1 < 1 && _preload.WithinBudget(factoryType1)) {
    action->at(StateBuildMap.at(factoryType1)) = 1;
    _preload.DeductBudget(factoryType1);
  }

  auto factoryType2 = gamedef.GetBuildFrom(unitType2);
  int numFactory2 = _preload.NumUnit(factoryType2, true);
  if (numFactory2 < 1 && _preload.WithinBudget(factoryType2)) {
    action->at(StateBuildMap.at(factoryType2)) = 1;
    _preload.DeductBudget(factoryType2);
  }

  const auto& myTroops = _preload.MyTroops();
  const auto& myArmy = _preload.MyArmy();

  int numBuiltFactory1 = _preload.NumUnit(factoryType1, false);
  int numArmy1 = _preload.NumUnit(unitType1, true);
  if (numBuiltFactory1 >= 1 && _preload.WithinBudget(unitType1) &&
      numArmy1 < minArmy1) {
    action->at(StateBuildMap.at(unitType1)) = 1;
    _preload.DeductBudget(unitType1);
  }

  int numBuiltFactory2 = _preload.NumUnit(factoryType2, false);
  int numArmy2 = _preload.NumUnit(unitType2, true);
  bool needMoreArmy = (maxArmy < 0 || (int)myArmy.size() < maxArmy);
  if (needMoreArmy && numBuiltFactory2 >= 1 &&
      _preload.WithinBudget(unitType2) &&
      (numArmy2 < minArmy2 || numArmy1 >= minArmy1)) {
    action->at(StateBuildMap.at(unitType2)) = 1;
    _preload.DeductBudget(unitType2);
  }

  if ((int)myArmy.size() >= minArmy1 + minArmy2 || myTroops[RESOURCE].empty() ||
      myTroops[PEASANT].empty()) {
    if (_preload.EnemyTroops()[TOWN_HALL].size() == 0) {
      action->at(STATE_SCOUT) = 1;
    }
    action->at(STATE_ATTACK_BASE) = 1;
  }
}

void MCStrategyActor::RushAgainstDragon(std::vector<int64_t>* action) {
  assert(_preload.Ready());

  // const auto& myArmy = _preload.MyArmy();
  // const auto& myTroops = _preload.MyTroops();
  const auto& enemyTroops = _preload.EnemyTroops();

  if (enemyTroops[DRAGON].size() > 0) {
    if (enemyTroops[TOWN_HALL].size() == 0) {
      action->at(STATE_SCOUT) = 1;
    }
    action->at(STATE_ATTACK_BASE) = 1;
  }
}

void MCStrategyActor::BuildArmy(const GameDef& gamedef,
                                UnitType unitType,
                                std::vector<int64_t>* action) {
  assert(_preload.Ready());

  auto factoryType = gamedef.GetBuildFrom(unitType);
  int numFactory = _preload.NumUnit(factoryType, true);
  if (numFactory < 1 && _preload.WithinBudget(factoryType)) {
    action->at(StateBuildMap.at(factoryType)) = 1;
    _preload.DeductBudget(factoryType);
  }

  int numBuiltFactory = _preload.NumUnit(factoryType, false);
  if (numBuiltFactory >= 1 && _preload.WithinBudget(unitType)) {
    action->at(StateBuildMap.at(unitType)) = 1;
    _preload.DeductBudget(unitType);
  }
}

void MCStrategyActor::AttackBase(std::vector<int64_t>* action) {
  if (_preload.EnemyTroops()[TOWN_HALL].size() == 0) {
    action->at(STATE_SCOUT) = 1;
  }
  action->at(STATE_ATTACK_BASE) = 1;
}

void MCStrategyActor::Expand(std::vector<int64_t>* action) {
  if (_preload.NumUnit(RESOURCE, false) <=
      _preload.NumTownHallWithResource(true)) {
    action->at(STATE_SCOUT) = 1;
  }
  if (_preload.WithinBudget(TOWN_HALL)) {
    action->at(STATE_BUILD_TOWN_HALL) = 1;
  }
}

void MCStrategyActor::BuildTower(std::vector<int64_t>* action) {
  if (_preload.WithinBudget(GUARD_TOWER)) {
    action->at(STATE_BUILD_GUARD_TOWER) = 1;
    _preload.DeductBudget(GUARD_TOWER);
  }
}

void MCStrategyActor::BuildPeasant(std::vector<int64_t>* action) {
  // hack to prevent congestion
  if (_preload.WithinBudget(PEASANT) && _preload.NumUnit(PEASANT, true) < 20) {
    action->at(STATE_BUILD_PEASANT) = 1;
    _preload.DeductBudget(PEASANT);
  }
}

void MCStrategyActor::ConservativeScout(std::vector<int64_t>* action) {
  const auto& enemyTroops = _preload.EnemyTroops();
  const std::vector<UnitType> targets = {DRAGON,
                                         SWORDMAN,
                                         SPEARMAN,
                                         CAVALRY,
                                         ARCHER,
                                         WORKSHOP,
                                         BARRACK,
                                         BLACKSMITH,
                                         STABLE};
  bool info_obtained = false;

  for (auto ut : targets) {
    if (enemyTroops.at(ut).size() > 0) {
      info_obtained = true;
      // std::cout << "info obtained, enemy has: " << ut << std::endl;
      break;
    }
  }
  if (!info_obtained) {
    action->at(STATE_SCOUT) = 1;
  }
}

void MCStrategyActor::Scout(std::vector<int64_t>* action) {
  action->at(STATE_SCOUT) = 1;
}

void MCStrategyActor::Prepare(std::vector<int64_t>* action) {
  action->at(STATE_GATHER) = 1;
  needDefend_ = DefendIfNeeded(action);
  BuildPeasantIfNeeded(optimalNumPeasant_, action);
}

// this function should only be used by trained AIs
bool MCStrategyActor::ActByStrategy(const GameEnv& env,
                                    AI* ai,
                                    RTSAction* action) {
  const std::vector<int64_t>& strategy = action->GetAction();
  assert((int)strategy.size() == NUM_STRATEGY);
  std::vector<int64_t> actionState(GameDef::GetNumAction(), 0);
  const auto gamedef = env.GetGameDef();

  // auto gather for trained AI
  actionState.at(STATE_GATHER) = 1;
  // auto defend for traind AI
  needDefend_ = DefendIfNeeded(&actionState);

  // build, for now these are in one softmax
  if (strategy[STRATEGY_BUILD_SWORDMAN]) {
    BuildArmy(gamedef, SWORDMAN, &actionState);
  }
  if (strategy[STRATEGY_BUILD_SPEARMAN]) {
    BuildArmy(gamedef, SPEARMAN, &actionState);
  }
  if (strategy[STRATEGY_BUILD_CAVALRY]) {
    BuildArmy(gamedef, CAVALRY, &actionState);
  }
  if (strategy[STRATEGY_BUILD_ARCHER]) {
    BuildArmy(gamedef, ARCHER, &actionState);
  }
  if (strategy[STRATEGY_BUILD_DRAGON]) {
    BuildArmy(gamedef, DRAGON, &actionState);
  }
  if (strategy[STRATEGY_BUILD_CATAPULT]) {
    BuildArmy(gamedef, CATAPULT, &actionState);
  }
  if (strategy[STRATEGY_BUILD_TOWER]) {
    BuildTower(&actionState);
  }
  if (strategy[STRATEGY_BUILD_PEASANT]) {
    BuildPeasant(&actionState);
  }
  // else {
  //   assert(strategy[STRATEGY_IDLE_BUILD]);
  // }

  // scout
  if (strategy[STRATEGY_SCOUT]) {
    Scout(&actionState);
  }
  // else {
  //   assert(strategy[STRATEGY_IDLE_SCOUT]);
  // }

  if (strategy[STRATEGY_ATTACK_BASE]) {
    AttackBase(&actionState);
  }
  // else {
  //   assert(strategy[STRATEGY_IDLE_ATTACK_BASE]);
  // }

  action->SetAction(std::move(actionState));
  return ActByState(env, ai, action);
}
