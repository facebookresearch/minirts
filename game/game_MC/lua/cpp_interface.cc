// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#include "cpp_interface.h"

void RTSMapGenerator::Init(const std::string& lua_files) {
  init(lua_files, "map_generator.lua");
}

// void RTSMapGenerator::Generate(RTSMap& map, int num_players, int seed) {
//   std::cout << "geneate " << std::endl;
//   Invoke<void>("rts_map_generator", "generate", map, num_players, seed);
// }

void RTSMapGenerator::GenerateRandom(
    RTSMap& map,
    int num_players,
    int seed,
    bool no_terrain) {
  Invoke<void>(
      "rts_map_generator",
      "generate_random",
      map,
      num_players,
      seed,
      no_terrain);
}

void RTSUnitGenerator::Init(const std::string& lua_files) {
  init(lua_files, "unit_generator.lua");
}

// void RTSUnitGenerator::Generate(
//     RTSMap* map,
//     int num_players,
//     int seed,
//     CmdReceiver* cmd_receiver) {
//   auto proxy = StateProxy{map, cmd_receiver};
//   Invoke<void>("rts_unit_generator", "generate", proxy, num_players, seed);
// }

void RTSUnitGenerator::GenerateRandom(
    RTSMap* map,
    CmdReceiver* cmd_receiver,
    int num_players,
    int seed,
    int resource,
    int resource_dist,
    int num_resources,
    bool fair,
    int num_peasants,
    int num_extra_units) {
  auto proxy = StateProxy{map, cmd_receiver};
  Invoke<void>(
      "rts_unit_generator",
      "generate_random",
      proxy,
      num_players,
      seed,
      resource,
      resource_dist,
      num_resources,
      fair,
      num_peasants,
      num_extra_units);
}

void RTSUnitFactory::Init(const std::string& lua_files) {
  init(lua_files, "unit_factory.lua");
}

UnitTemplate RTSUnitFactory::InitResource() {
  UnitTemplate ret;
  Invoke("rts_unit_factory", "init_resource", &ret);
  return ret;
}

UnitTemplate RTSUnitFactory::InitPeasant() {
  UnitTemplate ret;
  Invoke("rts_unit_factory", "init_peasant", &ret);
  return ret;
}

UnitTemplate RTSUnitFactory::InitSwordman() {
  UnitTemplate ret;
  Invoke("rts_unit_factory", "init_swordman", &ret);
  return ret;
}

UnitTemplate RTSUnitFactory::InitSpearman() {
  UnitTemplate ret;
  Invoke("rts_unit_factory", "init_spearman", &ret);
  return ret;
}

UnitTemplate RTSUnitFactory::InitCavalry() {
  UnitTemplate ret;
  Invoke("rts_unit_factory", "init_cavalry", &ret);
  return ret;
}

UnitTemplate RTSUnitFactory::InitDragon() {
  UnitTemplate ret;
  Invoke("rts_unit_factory", "init_dragon", &ret);
  return ret;
}

UnitTemplate RTSUnitFactory::InitArcher() {
  UnitTemplate ret;
  Invoke("rts_unit_factory", "init_archer", &ret);
  return ret;
}

UnitTemplate RTSUnitFactory::InitCatapult() {
  UnitTemplate ret;
  Invoke("rts_unit_factory", "init_catapult", &ret);
  return ret;
}

UnitTemplate RTSUnitFactory::InitBarrack() {
  UnitTemplate ret;
  Invoke("rts_unit_factory", "init_barrack", &ret);
  return ret;
}

UnitTemplate RTSUnitFactory::InitBlacksmith() {
  UnitTemplate ret;
  Invoke("rts_unit_factory", "init_blacksmith", &ret);
  return ret;
}

UnitTemplate RTSUnitFactory::InitStable() {
  UnitTemplate ret;
  Invoke("rts_unit_factory", "init_stable", &ret);
  return ret;
}

UnitTemplate RTSUnitFactory::InitWorkshop() {
  UnitTemplate ret;
  Invoke("rts_unit_factory", "init_workshop", &ret);
  return ret;
}

UnitTemplate RTSUnitFactory::InitAviary() {
  UnitTemplate ret;
  Invoke("rts_unit_factory", "init_aviary", &ret);
  return ret;
}

UnitTemplate RTSUnitFactory::InitArchery() {
  UnitTemplate ret;
  Invoke("rts_unit_factory", "init_archery", &ret);
  return ret;
}

UnitTemplate RTSUnitFactory::InitGuardTower() {
  UnitTemplate ret;
  Invoke("rts_unit_factory", "init_guard_tower", &ret);
  return ret;
}

UnitTemplate RTSUnitFactory::InitTownHall() {
  UnitTemplate ret;
  Invoke("rts_unit_factory", "init_town_hall", &ret);
  return ret;
}

void reg_cpp_interfaces(const std::string& lua_files) {
  RTSMapGenerator::Init(lua_files);
  RTSUnitGenerator::Init(lua_files);
  RTSUnitFactory::Init(lua_files);
}
