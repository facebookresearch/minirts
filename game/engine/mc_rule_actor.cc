// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#include "mc_rule_actor.h"
#include "engine/utils.h"

const Unit* GetTargetUnit(const Unit* attacker,
                          const std::vector<const Unit*>& enemy_units,
                          const GameDef& gamedef,
                          const CmdReceiver& receiver,
                          float d_sqr) {
  auto attacker_type = attacker->GetUnitType();
  auto unit_type_to_attack_order =
      gamedef.unit(attacker_type).GetUnitType2AttackOrder();
  const PointF& attacker_p = attacker->GetPointF();

  const Unit* target = nullptr;
  int target_order = std::numeric_limits<int>::max();

  for (auto enemy : enemy_units) {
    if (d_sqr > 0 && PointF::L2Sqr(enemy->GetPointF(), attacker_p) > d_sqr) {
      continue;
    }

    auto enemy_type = enemy->GetUnitType();
    int attack_order = unit_type_to_attack_order[enemy_type];
    if (attack_order < target_order) {
      target = enemy;
      target_order = attack_order;
      continue;
    }

    if (target == nullptr && gamedef.CanAttack(attacker_type, enemy_type)) {
      assert(target_order == attack_order);
      target = enemy;
      continue;
    }
  }
  // From peasant, attack those that construct something
  if (target != nullptr && target->GetUnitType() == PEASANT) {
    auto new_target =
        closestUnitInCmd(enemy_units, attacker_p, receiver, BUILD);
    if (new_target != nullptr) {
      return new_target;
    }
  }
  return target;
}

bool FindBuilderAndPlace(const GameEnv& env,
                         const std::vector<const Unit*>& peasants,
                         const PointF& center,
                         float building_l1_radius,
                         const std::set<PointF>& place_taken,
                         const std::map<UnitId, CmdBPtr>& pending_cmds,
                         const CmdReceiver& receiver,
                         const Unit** builder,
                         PointF* place) {
  // first find a place
  PointF p;
  const UnitTemplate& builder_def = env.GetGameDef().unit(PEASANT);
  bool have_place = env.FindEmptyPlaceNearby(
      builder_def, center, building_l1_radius, place_taken, &p);
  if (!have_place || p.IsInvalid()) {
    return false;
  }
  // next find a builder
  const Unit* worker =
      closestUnitWithNoPendingCmd(peasants, p, pending_cmds, receiver, BUILD);
  if (worker == nullptr) {
    return false;
  }
  *builder = worker;
  *place = p;
  return true;
}

bool MCRuleActor::ActByState(const GameEnv& env, AI* ai, RTSAction* action) {
  const auto game_status = env.GetGameStatus();
  if (game_status != ACTIVE_STATUS) {
    // don't issue any commands if the game is frozen
    return true;
  }
  const Player& player = env.GetPlayer(_player_id);
  const float building_l1_radius = 3.0;  // TODO: put into config
  const auto& action_state = action->GetAction();
  const GameDef& gamedef = env.GetGameDef();
  AssignedCmds& cmds = action->cmds();

  assert(static_cast<int>(action_state.size()) == GameDef::GetNumAction());

  // select town_hall
  auto my_town_halls = _preload.MyTownHalls();
  assert((int)my_town_halls.size() > 0);
  int focused_town_hall = action->GetFocusedTownHall();
  if (focused_town_hall >= (int)my_town_halls.size()) {
    focused_town_hall = 0;
  }
  const TownHall& town_hall = my_town_halls[focused_town_hall];

  // Make guard towers auto attack
  SetTowerAutoAttack(env, &cmds);
  PeasantDefend(env, &cmds);

  // build new units
  build_units(env,
              town_hall,
              action_state,
              building_l1_radius,
              ai->maxBuildQueueSize,
              &(ai->buildQueue),
              &(ai->rng),
              &cmds);

  // the following ordering should not be randomized
  if (action_state[STATE_SCOUT]) {
    scout(env, player, &cmds);
  }

  if (action_state[STATE_DEFEND]) {
    defend(gamedef, &cmds);
  }

  if (action_state[STATE_ATTACK_IN_RANGE]) {
    attack_in_range(gamedef, &cmds);
  }

  if (action_state[STATE_ATTACK_BASE]) {
    attack_base(gamedef, &cmds);
  }

  if (action_state[STATE_ATTACK_PEASANT_BASE]) {
    attack_peasant_base(gamedef, &cmds);
  }

  if (action_state[STATE_ATTACK_TOWER_RUSH]) {
    attack_tower_rush(env, &cmds, building_l1_radius * 2);
  }

  // gather should have lowest priority
  if (action_state[STATE_GATHER]) {
    gather(ai->resourceScale(), &cmds);
  }

  return true;
}

bool MCRuleActor::SetTowerAutoAttack(const GameEnv& env,
                                     AssignedCmds* assigned_cmds) {
  const GameDef& gamedef = env.GetGameDef();
  const auto& my_towers = _preload.MyTroops()[GUARD_TOWER];
  float attack_range = gamedef.unit(GUARD_TOWER).GetProperty()._att_r;
  const auto& enemy_in_range_targets = _preload.EnemyInRangeTargets();

  for (const Unit* t : my_towers) {
    std::vector<const Unit*> reachable_targets =
        filterByDistance(enemy_in_range_targets, t->GetPointF(), attack_range);
    const Unit* target =
        GetTargetUnit(t, reachable_targets, gamedef, _receiver, -1.0);
    if (target != nullptr) {
      store_cmd(t, _A(target->GetId()), assigned_cmds);
    }
  }

  return true;
}

bool MCRuleActor::PeasantDefend(const GameEnv& env,
                                AssignedCmds* assigned_cmds) {
  const GameDef& gamedef = env.GetGameDef();
  float attack_range = gamedef.unit(GUARD_TOWER).GetProperty()._att_r;
  const auto& enemy_in_range_targets = _preload.EnemyInRangeTargets();

  // peasant attack enemy peasant nearby, [TODO] move to new function
  for (const Unit* p : _preload.MyTroops()[PEASANT]) {
    if (!IsIdleOrGather(_receiver, *p)) {
      continue;
    }
    std::vector<const Unit*> reachable_targets = filterByDistance(
        enemy_in_range_targets, p->GetPointF(), attack_range + 3);
    for (const Unit* ep : reachable_targets) {
      if (ep->GetUnitType() != PEASANT) {
        continue;
      }
      if (InCmd(_receiver, *ep, BUILD) ||
          PointF::L2Sqr(ep->GetPointF(), p->GetPointF()) <= 9) {
        store_cmd(p, _A(ep->GetId()), assigned_cmds);
      }
    }
  }
  return true;
}

// entry point for building all kinds of units
// also handle affordability and queueing
void MCRuleActor::build_units(const GameEnv& env,
                              const TownHall& focus_town_hall,
                              const std::vector<int64_t>& action_state,
                              const float building_l1_radius,
                              size_t max_build_queue_size,
                              std::list<UnitType>* build_queue,
                              std::mt19937* rng,
                              AssignedCmds* assigned_cmds) {
  // std::cout << ">>>>enter build units, queue size: "
  //           << build_queue->size() << std::endl;
  assert(build_queue->size() <= max_build_queue_size);
  while (build_queue->size() > 0) {
    UnitType unit_type = build_queue->front();
    if (!_preload.Affordable(unit_type)) {
      break;
    }
    build_queue->pop_front();
    // std::cout << "build from queue" << std::endl;
    build_unit(
        unit_type, env, focus_town_hall, building_l1_radius, assigned_cmds);
    // std::cout << "done build from queue" << std::endl;
  }

  std::vector<UnitType> traverse_order;
  for (int type = 0; type < NUM_MINIRTS_UNITTYPE; ++type) {
    traverse_order.push_back(static_cast<UnitType>(type));
  }
  // shuffle the build order
  if (rng != nullptr) {
    std::random_shuffle(traverse_order.begin(),
                        traverse_order.end(),
                        [=](int max) { return (*rng)() % max; });
  }

  for (UnitType unit_type : traverse_order) {
    // queue is full, ignore all subsequent build command
    if (build_queue->size() >= max_build_queue_size) {
      // std::cout << "queue is full, ignore build cmd, "
      //           << build_queue->size() << std::endl;
      break;
    }

    // queue is not empty, block all future build cmd
    if (build_queue->size() > 0) {
      // std::cout << "queue is not empty, enqueue:" << unit_type << std::endl;
      build_queue->push_back(unit_type);
      continue;
    }

    auto unit2cmd = StateBuildMap.find(unit_type);
    // the unit is not buildable, such as resource, archery
    if (unit2cmd == StateBuildMap.end() || !action_state.at(unit2cmd->second)) {
      continue;
    }

    if (!_preload.Affordable(unit_type)) {
      // std::cout << "cannot afford, enqueue: " << unit_type << std::endl;
      build_queue->push_back(unit_type);
      continue;
    }

    // queue is clear and build cmd is affordable
    build_unit(
        unit_type, env, focus_town_hall, building_l1_radius, assigned_cmds);
  }
  // std::cout << "<<<<exit build units: queue size:"
  //           << build_queue->size() << std::endl;
}

// this function should not worry about build queue
// the caller should make sure that money is sufficient
bool MCRuleActor::build_unit(UnitType unit_type,
                             const GameEnv& env,
                             const TownHall& focus_town_hall,
                             float building_l1_radius,
                             AssignedCmds* assigned_cmds) {
  assert(_preload.Affordable(unit_type));

  if (unit_type == TOWN_HALL) {
    return build_town_hall(env, building_l1_radius, assigned_cmds);
  }

  const GameDef& gamedef = env.GetGameDef();
  const auto& my_troops = _preload.MyTroops();
  const UnitType& build_from = gamedef.unit(unit_type).GetBuildFrom();
  if (gamedef.IsUnitTypeBuilding(build_from)) {
    // build from a building
    const Unit* u = nullptr;
    if (unit_type == PEASANT) {
      u = focus_town_hall.GetTownHall();
    } else {
      u = PickFirstIdle(my_troops[build_from], *assigned_cmds, _receiver);
    }
    if (u != nullptr) {
      assert(_preload.BuildIfAffordable(unit_type));
      store_cmd(u, _B(unit_type), assigned_cmds);
    }
  } else {
    // build from peasant, a building
    PointF new_p;
    const Unit* builder = nullptr;
    bool can_build = FindBuilderAndPlace(env,
                                         my_troops[PEASANT],
                                         focus_town_hall.GetPointF(),
                                         building_l1_radius,
                                         _place_taken,
                                         *assigned_cmds,
                                         _receiver,
                                         &builder,
                                         &new_p);
    if (!can_build) {
      return false;
    }
    // std::cout << "build: " << unit_type << std::endl;
    assert(_preload.BuildIfAffordable(unit_type));
    store_cmd(builder, _B(unit_type, new_p), assigned_cmds);
    _place_taken.insert(new_p);
  }
  return true;
}

bool MCRuleActor::build_town_hall(const GameEnv& env,
                                  float building_l1_radius,
                                  AssignedCmds* assigned_cmds) {
  const auto& my_troops = _preload.MyTroops();
  if (my_troops[PEASANT].size() == 0) {
    return false;
  }

  float dsqr_bound = (building_l1_radius + 3) * (building_l1_radius + 3);
  float closest_resource_dsqr = 1e4;
  const Unit* closest_resource = nullptr;

  for (const Unit* resource : my_troops[RESOURCE]) {
    // check if there is a town_hall around resource already
    PointF resource_loc = resource->GetPointF();
    float dsqr = 1e10;
    closestUnit(my_troops[TOWN_HALL], resource_loc, dsqr, &dsqr);
    if (dsqr < dsqr_bound) {
      // have base close to this resource
      continue;
    }
    if (dsqr < closest_resource_dsqr) {
      closest_resource_dsqr = dsqr;
      closest_resource = resource;
    }
  }

  if (closest_resource == nullptr) {
    return false;
  }

  PointF new_p;
  const Unit* builder = nullptr;
  bool can_build = FindBuilderAndPlace(env,
                                       my_troops[PEASANT],
                                       closest_resource->GetPointF(),
                                       building_l1_radius,
                                       _place_taken,
                                       *assigned_cmds,
                                       _receiver,
                                       &builder,
                                       &new_p);
  if (!can_build) {
    return false;
  }
  assert(_preload.BuildIfAffordable(TOWN_HALL));
  store_cmd(builder, _B(TOWN_HALL, new_p), assigned_cmds);
  _place_taken.insert(new_p);

  return true;
}

bool MCRuleActor::gather(float resource_scale, AssignedCmds* assigned_cmds) {
  const auto& my_troops = _preload.MyTroops();

  for (const Unit* u : my_troops[PEASANT]) {
    if (!IsIdle(_receiver, *u) || UnitHasCmd(u->GetId(), *assigned_cmds)) {
      continue;
    }
    // Gather!
    auto town_hall = _preload.GetDestinationForGather();
    if (town_hall == nullptr) {
      return false;
    }
    town_hall->AddPeasant();
    auto town_hall_id = town_hall->GetId();
    auto resource_id = town_hall->GetResourceId();
    if (resource_id != INVALID) {
      // not out of resource
      store_cmd(
          u, _G(town_hall_id, resource_id, resource_scale), assigned_cmds);
    }
  }
  return true;
}

bool MCRuleActor::attack(const GameDef& gamedef,
                         const std::vector<const Unit*>& my_units,
                         const std::vector<const Unit*>& enemy_units,
                         float d_sqr,
                         AssignedCmds* assigned_cmds) {
  if (my_units.empty() || enemy_units.empty()) {
    return false;
  }

  bool cmd_issued = false;
  for (const Unit* my_unit : my_units) {
    // skip if unit has pending cmd
    if (assigned_cmds->find(my_unit->GetId()) != assigned_cmds->end()) {
      continue;
    }

    const Unit* target =
        GetTargetUnit(my_unit, enemy_units, gamedef, _receiver, d_sqr);
    if (target != nullptr) {
      store_cmd(my_unit, _A(target->GetId()), assigned_cmds);
      cmd_issued = true;
    }
  }
  return cmd_issued;
}

bool MCRuleActor::attack_base(const GameDef& gamedef,
                              AssignedCmds* assigned_cmds) {
  auto my_units = _preload.MyArmy();
  auto enemy_units = _preload.EnemyBaseTargets();
  return attack(gamedef, my_units, enemy_units, -1.0, assigned_cmds);
}

bool MCRuleActor::attack_peasant_base(const GameDef& gamedef,
                                      AssignedCmds* assigned_cmds) {
  auto my_units = _preload.MyTroops()[PEASANT];
  auto enemy_units = _preload.EnemyAllUnits();
  // first attack units, if any
  if (enemy_units.empty()) {
    auto enemy_buildings = _preload.EnemyBaseTargets();
    return attack(gamedef, my_units, enemy_buildings, -1.0, assigned_cmds);
  }
  return attack(gamedef, my_units, enemy_units, -1.0, assigned_cmds);
}

bool MCRuleActor::attack_in_range(const GameDef& gamedef,
                                  AssignedCmds* assigned_cmds) {
  auto my_units = _preload.MyArmy();
  auto enemy_units = _preload.EnemyInRangeTargets();
  return attack(gamedef, my_units, enemy_units, 200, assigned_cmds);
}

bool MCRuleActor::attack_tower_rush(const GameEnv& env,
                                    AssignedCmds* assigned_cmds,
                                    const float building_l1_radius) {
  // TODO: merge this function with build_unit

  auto enemy_buildings = _preload.EnemyBaseTargets();
  assert(!enemy_buildings.empty());
  const auto center = enemy_buildings.front()->GetPointF();

  const auto& my_peasants = _preload.MyTroops()[PEASANT];
  PointF new_p;
  const Unit* builder = nullptr;
  const bool can_build = FindBuilderAndPlace(env,
                                             my_peasants,
                                             center,
                                             building_l1_radius,
                                             _place_taken,
                                             *assigned_cmds,
                                             _receiver,
                                             &builder,
                                             &new_p);
  if (!can_build) {
    return false;
  }

  assert(_preload.BuildIfAffordable(GUARD_TOWER));
  store_cmd(builder, _B(GUARD_TOWER, new_p), assigned_cmds);
  _place_taken.insert(new_p);
  return true;
}

bool MCRuleActor::defend(const GameDef& gamedef, AssignedCmds* assigned_cmds) {
  auto my_units = _preload.MyArmy();
  auto enemy_units = _preload.EnemyDefendTargets();
  return attack(gamedef, my_units, enemy_units, -1.0, assigned_cmds);
}

const Unit* find_scouter(const std::vector<std::vector<const Unit*>>& troops,
                         const PointF& destination,
                         const AssignedCmds& pending_cmds,
                         const CmdReceiver& receiver) {
  const std::vector<UnitType>& scout_order = {
      DRAGON, PEASANT, SPEARMAN, CAVALRY, SWORDMAN, ARCHER};
  const Unit* scouter = nullptr;

  for (const UnitType& target_type : scout_order) {

    const auto& target_units = troops[target_type];
    scouter = closestUnitWithNoPendingCmd(
        target_units, destination, pending_cmds, receiver, BUILD);
    if (scouter != nullptr) {
      break;
    }
  }
  return scouter;
}

bool MCRuleActor::scout(const GameEnv& env,
                        const Player& player,
                        AssignedCmds* assigned_cmds) {
  const auto& map = env.GetMap();
  const int map_x_size = map.GetXSize();
  const int map_y_size = map.GetYSize();
  assert(map_x_size == map_y_size);
  const int margin = 4;

  auto town_hall_loc = _preload.MyTownHalls().at(0).GetPointF();
  int town_hall_x = town_hall_loc.x;
  int town_hall_y = town_hall_loc.y;
  int offset;
  if (town_hall_x <= map_x_size / 2 && town_hall_y <= map_y_size / 2) {
    offset = 0;
  } else if (town_hall_x <= map_x_size / 2 && town_hall_y > map_y_size / 2) {
    offset = 1;
  } else if (town_hall_x > map_x_size / 2 && town_hall_y > map_y_size / 2) {
    offset = 2;
  } else {
    offset = 3;
  }

  PointF destination;
  for (int d = 2; d < map_x_size / 2; d += margin) {
    for (int i = 0; i < 4; ++i) {
      int direction = (i + offset) % 4;
      int x, y;
      int (*x_inc)(int) = [](int x) { return x; };
      int (*y_inc)(int) = [](int y) { return y; };

      if (direction == 0) {
        // go down
        x = d;
        y = d;
        y_inc = [](int y) { return y + margin; };
      } else if (direction == 1) {
        // go left
        x = d;
        y = map_y_size - d;
        x_inc = [](int x) { return x + margin; };
      } else if (direction == 2) {
        // go up
        x = map_x_size - d;
        y = map_y_size - d;
        y_inc = [](int y) { return y - margin; };
      } else {
        // go left
        x = map_x_size - d;
        y = d;
        x_inc = [](int x) { return x - margin; };
      }

      while (x <= map_x_size - d && x > 0 && y <= map_y_size - d && y > 0) {
        Coord c(x, y);
        Loc loc = map.GetLoc(c);
        const Fog& f = player.GetFog(loc);
        x = x_inc(x);
        y = y_inc(y);
        if (f.HasSeenTerrain()) {
          continue;
        }

        destination = PointF(c);
        goto DESTINATION_FOUND;
      }
    }
  }
  return false;

DESTINATION_FOUND:
  // find unit for scouting
  const auto& my_troops = _preload.MyTroops();
  const Unit* scouter =
      find_scouter(my_troops, destination, *assigned_cmds, _receiver);
  if (scouter == nullptr) {
    return false;
  }
  store_cmd(scouter, _M(destination), assigned_cmds);
  return true;
}
