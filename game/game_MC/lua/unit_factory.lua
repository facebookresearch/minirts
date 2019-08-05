-- Copyright (c) Facebook, Inc. and its affiliates.
-- All rights reserved.
--
-- This source code is licensed under the license found in the
-- LICENSE file in the root directory of this source tree.
--
rts_unit_factory = {}

--[[
--Cooldown is specified by a vector of 4 numbers in the following order:
--[CD_MOVE, CD_ATTACK, CD_GATHER, CD_BUILD]
]]

function __create_build_skill(unit_type, hotkey)
  local skill = BuildSkill.new()
  skill:set_unit_type(unit_type)
  skill:set_hotkey(hotkey)
  return skill
end

function __create_unit_template(
    cost, hp, defense, speed, att, att_r, vis_r,
    cds, allowed_cmds, attr, attack_multiplier, attack_order,
    cant_move_over, build_from, build_skills)

  local up = UnitProperty.new()
  up:set_hp(hp)
  up:set_max_hp(hp)
  up:set_speed(speed)
  up:set_def(defense)
  up:set_attr(attr)
  up:set_att(att)
  up:set_att_r(att_r)
  up:set_vis_r(vis_r)

  for i = 1, CDType.NUM_COOLDOWN do
    local cd = Cooldown.new()
    cd:set(cds[i])
    up:set_cooldown(i - 1, cd)
  end

  local ut = UnitTemplate.new()
  ut:set_property(up)
  ut:set_build_cost(cost)

  for i = 1, #allowed_cmds do
    ut:add_allowed_cmd(allowed_cmds[i])
  end
  for i = 1, #attack_multiplier do
    local ty = attack_multiplier[i][1]
    local mult = attack_multiplier[i][2]
    ut:set_attack_multiplier(ty, mult)
  end
  for i = 1, #attack_order do
    local ao = attack_order[i]
    ut:append_attack_order(ao)
  end
  for i = 1, #cant_move_over do
    ut:add_cant_move_over(cant_move_over[i])
  end
  for i = 1, #build_skills do
    ut:add_build_skill(build_skills[i])
  end
  ut:set_build_from(build_from)
  return ut
end

function rts_unit_factory.init_resource()
  -- stationary, high hp, invisible
  local ut = __create_unit_template(
    --[[cost]]100000,
    --[[hp]]1000,
    --[[defense]]1000,
    --[[speed]]0,
    --[[att]]0,
    --[[att_r]]0,
    --[[vis_r]]0,
    --[[cds]]{0, 0, 0, 0},
    --[[allowed_cmds]]{},
    --[[attr]]UnitAttr.ATTR_INVULNERABLE,
    --[[attack_multiplier]]{},
    --[[attack_order]]{},
    --[[cant_move_over]]{},
    --[[build_from]]-1,
    --[[build_skills]]{}
  )
  return ut
end

function rts_unit_factory.init_peasant()
  -- worker unit, can't attack, only can build
  local allowed_cmds = {CmdType.MOVE, CmdType.ATTACK, CmdType.BUILD, CmdType.GATHER}
  local attack_multiplier = {
    {UnitType.PEASANT, 1.0},
    {UnitType.SWORDMAN, 0.7},
    {UnitType.SPEARMAN, 0.7},
    {UnitType.CAVALRY, 0.7},
    {UnitType.DRAGON, 0.7},
    {UnitType.ARCHER, 0.7},
    {UnitType.CATAPULT, 0.7},
    {UnitType.TOWN_HALL, 2.0},
    {UnitType.BARRACK, 2.0},
    {UnitType.STABLE, 2.0},
    {UnitType.BLACKSMITH, 2.0},
    {UnitType.WORKSHOP, 2.0},
    {UnitType.AVIARY, 2.0},
    {UnitType.ARCHERY, 2.0},
    {UnitType.GUARD_TOWER, 1.5},
  }
  local attack_order = {
    UnitType.SWORDMAN,
    UnitType.SPEARMAN,
    UnitType.CAVALRY,
    UnitType.DRAGON,
    UnitType.ARCHER,
    UnitType.CATAPULT,
    UnitType.GUARD_TOWER,
    UnitType.PEASANT,
    UnitType.TOWN_HALL,
  }
  local cant_move_over = {Terrain.ROCK, Terrain.WATER}
  local build_skills = {
    __create_build_skill(UnitType.TOWN_HALL, "h"),
    __create_build_skill(UnitType.BARRACK, "b"),
    __create_build_skill(UnitType.BLACKSMITH, "s"),
    __create_build_skill(UnitType.STABLE, "l"),
    __create_build_skill(UnitType.WORKSHOP, "w"),
    -- __create_build_skill(UnitType.AVIARY, "v"),
    -- __create_build_skill(UnitType.ARCHERY, "y"),
    __create_build_skill(UnitType.GUARD_TOWER, "t")
  }
  local ut = __create_unit_template(
    --[[cost]]50,
    --[[hp]]30,
    --[[defense]]0,
    --[[speed]]0.05,
    --[[att]]3,
    --[[att_r]]2,
    --[[vis_r]]7,
    --[[cds]]{0, 25, 120, 250},
    --[[allowed_cmds]]allowed_cmds,
    --[[attr]]UnitAttr.ATTR_NORMAL,
    --[[attack_multiplier]]attack_multiplier,
    --[[attack_order]]attack_order,
    --[[cant_move_over]]cant_move_over,
    --[[build_from]]UnitType.TOWN_HALL,
    --[[build_skills]]build_skills
  )
  return ut
end

function rts_unit_factory.init_swordman()
  local allowed_cmds = {CmdType.MOVE, CmdType.ATTACK}
  local attack_multiplier = {
    {UnitType.PEASANT, 1.0},
    {UnitType.SWORDMAN, 1.0},
    {UnitType.SPEARMAN, 1.5},
    {UnitType.CAVALRY, 0.5},
    {UnitType.DRAGON, 0.0},
    {UnitType.ARCHER, 1.0},
    {UnitType.CATAPULT, 1.0},
    {UnitType.TOWN_HALL, 1.0},
    {UnitType.BARRACK, 1.0},
    {UnitType.STABLE, 1.0},
    {UnitType.BLACKSMITH, 1.0},
    {UnitType.WORKSHOP, 1.0},
    {UnitType.AVIARY, 1.0},
    {UnitType.ARCHERY, 1.0},
    {UnitType.GUARD_TOWER, 1.0},
  }
  local attack_order = {
    UnitType.SPEARMAN,
    UnitType.GUARD_TOWER,
    UnitType.ARCHER,
    UnitType.CATAPULT,
    UnitType.SWORDMAN,
    UnitType.CAVALRY,
    UnitType.PEASANT,
    UnitType.TOWN_HALL
  }
  local cant_move_over = {Terrain.ROCK, Terrain.WATER}
  local build_skills = {}
  local ut = __create_unit_template(
    --[[cost]]100,
    --[[hp]]100,
    --[[defense]]1,
    --[[speed]]0.05,
    --[[att]]7,
    --[[att_r]]2,
    --[[vis_r]]7,
    --[[cds]]{0, 20, 0, 0},
    --[[allowed_cmds]]allowed_cmds,
    --[[attr]]UnitAttr.ATTR_NORMAL,
    --[[attack_multiplier]]attack_multiplier,
    --[[attack_order]]attack_order,
    --[[cant_move_over]]cant_move_over,
    --[[build_from]]UnitType.BLACKSMITH,
    --[[build_skills]]build_skills
  )
  return ut
end

function rts_unit_factory.init_spearman()
  local allowed_cmds = {CmdType.MOVE, CmdType.ATTACK}
  local attack_multiplier = {
    {UnitType.PEASANT, 1.0},
    {UnitType.SWORDMAN, 0.5},
    {UnitType.SPEARMAN, 1.0},
    {UnitType.CAVALRY, 1.5},
    {UnitType.DRAGON, 0.0},
    {UnitType.ARCHER, 1.0},
    {UnitType.CATAPULT, 1.0},
    {UnitType.TOWN_HALL, 1.0},
    {UnitType.BARRACK, 1.0},
    {UnitType.STABLE, 1.0},
    {UnitType.BLACKSMITH, 1.0},
    {UnitType.WORKSHOP, 1.0},
    {UnitType.AVIARY, 1.0},
    {UnitType.ARCHERY, 1.0},
    {UnitType.GUARD_TOWER, 1.0},
  }
  local attack_order = {
    UnitType.CAVALRY,
    UnitType.GUARD_TOWER,
    UnitType.ARCHER,
    UnitType.CATAPULT,
    UnitType.SPEARMAN,
    UnitType.SWORDMAN,
    UnitType.PEASANT,
    UnitType.TOWN_HALL
  }
  local cant_move_over = {Terrain.ROCK, Terrain.WATER}
  local build_skills = {}
  local ut = __create_unit_template(
    --[[cost]]100,
    --[[hp]]100,
    --[[defense]]1,
    --[[speed]]0.05,
    --[[att]]7,
    --[[att_r]]2,
    --[[vis_r]]7,
    --[[cds]]{0, 20, 0, 0},
    --[[allowed_cmds]]allowed_cmds,
    --[[attr]]UnitAttr.ATTR_NORMAL,
    --[[attack_multiplier]]attack_multiplier,
    --[[attack_order]]attack_order,
    --[[cant_move_over]]cant_move_over,
    --[[build_from]]UnitType.BARRACK,
    --[[build_skills]]build_skills
  )
  return ut
end

function rts_unit_factory.init_cavalry()
  local allowed_cmds = {CmdType.MOVE, CmdType.ATTACK}
  local attack_multiplier = {
    {UnitType.PEASANT, 1.0},
    {UnitType.SWORDMAN, 1.5},
    {UnitType.SPEARMAN, 0.5},
    {UnitType.CAVALRY, 1.0},
    {UnitType.DRAGON, 0.0},
    {UnitType.ARCHER, 1.0},
    {UnitType.CATAPULT, 1.0},
    {UnitType.TOWN_HALL, 1.0},
    {UnitType.BARRACK, 1.0},
    {UnitType.STABLE, 1.0},
    {UnitType.BLACKSMITH, 1.0},
    {UnitType.WORKSHOP, 1.0},
    {UnitType.AVIARY, 1.0},
    {UnitType.ARCHERY, 1.0},
    {UnitType.GUARD_TOWER, 1.0},
  }
  local attack_order = {
    UnitType.SWORDMAN,
    UnitType.GUARD_TOWER,
    UnitType.ARCHER,
    UnitType.CATAPULT,
    UnitType.CAVALRY,
    UnitType.SPEARMAN,
    UnitType.PEASANT,
    UnitType.TOWN_HALL
  }
  local cant_move_over = {Terrain.ROCK, Terrain.WATER}
  local build_skills = {}
  local ut = __create_unit_template(
    --[[cost]]100,
    --[[hp]]100,
    --[[defense]]1,
    --[[speed]]0.05,
    --[[att]]7,
    --[[att_r]]2,
    --[[vis_r]]7,
    --[[cds]]{0, 20, 0, 0},
    --[[allowed_cmds]]allowed_cmds,
    --[[attr]]UnitAttr.ATTR_NORMAL,
    --[[attack_multiplier]]attack_multiplier,
    --[[attack_order]]attack_order,
    --[[cant_move_over]]cant_move_over,
    --[[build_from]]UnitType.STABLE,
    --[[build_skills]]build_skills
  )
  return ut
end

function rts_unit_factory.init_archer()
  local allowed_cmds = {CmdType.MOVE, CmdType.ATTACK}
  local attack_multiplier = {
    {UnitType.PEASANT, 0.5},
    {UnitType.SWORDMAN, 0.5},
    {UnitType.SPEARMAN, 0.5},
    {UnitType.CAVALRY, 0.5},
    {UnitType.DRAGON, 2.0},
    {UnitType.ARCHER, 0.5},
    {UnitType.CATAPULT, 1.0},
    {UnitType.TOWN_HALL, 1.0},
    {UnitType.BARRACK, 1.0},
    {UnitType.STABLE, 1.0},
    {UnitType.BLACKSMITH, 1.0},
    {UnitType.WORKSHOP, 1.0},
    {UnitType.AVIARY, 1.0},
    {UnitType.ARCHERY, 1.0},
    {UnitType.GUARD_TOWER, 1.0},
  }
  local attack_order = {
    UnitType.DRAGON,
    UnitType.GUARD_TOWER,
    UnitType.ARCHER,
    UnitType.CATAPULT,
    UnitType.CAVALRY,
    UnitType.SWORDMAN,
    UnitType.SPEARMAN,
    UnitType.PEASANT,
    UnitType.TOWN_HALL
  }
  local cant_move_over = {Terrain.ROCK, Terrain.WATER}
  local build_skills = {}
  local ut = __create_unit_template(
    --[[cost]]100,
    --[[hp]]75,
    --[[defense]]0,
    --[[speed]]0.03,
    --[[att]]7,
    --[[att_r]]5,
    --[[vis_r]]7,
    --[[cds]]{0, 40, 0, 0},
    --[[allowed_cmds]]allowed_cmds,
    --[[attr]]UnitAttr.ATTR_NORMAL,
    --[[attack_multiplier]]attack_multiplier,
    --[[attack_order]]attack_order,
    --[[cant_move_over]]cant_move_over,
    --[[build_from]]UnitType.WORKSHOP,
    --[[build_skills]]build_skills
  )
  return ut
end

function rts_unit_factory.init_dragon()
  -- can fly, good for scouting
  local allowed_cmds = {CmdType.MOVE, CmdType.ATTACK}
  local attack_multiplier = {
    {UnitType.PEASANT, 1.0},
    {UnitType.SWORDMAN, 1.0},
    {UnitType.SPEARMAN, 1.0},
    {UnitType.CAVALRY, 1.0},
    {UnitType.DRAGON, 1.0},
    {UnitType.ARCHER, 0.5},
    {UnitType.CATAPULT, 1.0},
    {UnitType.TOWN_HALL, 1.0},
    {UnitType.BARRACK, 1.0},
    {UnitType.STABLE, 1.0},
    {UnitType.BLACKSMITH, 1.0},
    {UnitType.WORKSHOP, 1.0},
    {UnitType.AVIARY, 1.0},
    {UnitType.ARCHERY, 1.0},
    {UnitType.GUARD_TOWER, 1.0},
  }
  local attack_order = {
    UnitType.DRAGON,
    UnitType.GUARD_TOWER,
    UnitType.ARCHER,
    UnitType.CATAPULT,
    UnitType.CAVALRY,
    UnitType.SWORDMAN,
    UnitType.SPEARMAN,
    UnitType.PEASANT,
    UnitType.TOWN_HALL
  }
  local cant_move_over = {}
  local build_skills = {}
  local ut = __create_unit_template(
    --[[cost]]200,
    --[[hp]]100,
    --[[defense]]2,
    --[[speed]]0.04,
    --[[att]]10,
    --[[att_r]]5,
    --[[vis_r]]7,
    --[[cds]]{0, 40, 0, 0},
    --[[allowed_cmds]]allowed_cmds,
    --[[attr]]UnitAttr.ATTR_NORMAL,
    --[[attack_multiplier]]attack_multiplier,
    --[[attack_order]]attack_order,
    --[[cant_move_over]]cant_move_over,
    --[[build_from]]UnitType.WORKSHOP,
    --[[build_skills]]build_skills
  )
  return ut
end

function rts_unit_factory.init_catapult()
  local allowed_cmds = {CmdType.MOVE, CmdType.ATTACK}
  local attack_multiplier = {
    {UnitType.PEASANT, 1.0},
    {UnitType.SWORDMAN, 0.0},
    {UnitType.SPEARMAN, 0.0},
    {UnitType.CAVALRY, 0.0},
    {UnitType.DRAGON, 0.0},
    {UnitType.ARCHER, 0.0},
    {UnitType.CATAPULT, 1.0},
    {UnitType.TOWN_HALL, 2.0},
    {UnitType.BARRACK, 2.0},
    {UnitType.STABLE, 2.0},
    {UnitType.BLACKSMITH, 2.0},
    {UnitType.WORKSHOP, 2.0},
    {UnitType.AVIARY, 2.0},
    {UnitType.ARCHERY, 2.0},
    {UnitType.GUARD_TOWER, 2.0},
  }
  local attack_order = {
    UnitType.GUARD_TOWER,
    UnitType.TOWN_HALL
  }
  local cant_move_over = {Terrain.ROCK, Terrain.WATER}
  local build_skills = {}
  local ut = __create_unit_template(
    --[[cost]]100,
    --[[hp]]75,
    --[[defense]]2,
    --[[speed]]0.03,
    --[[att]]10,
    --[[att_r]]7,
    --[[vis_r]]7,
    --[[cds]]{0, 40, 0, 0},
    --[[allowed_cmds]]allowed_cmds,
    --[[attr]]UnitAttr.ATTR_NORMAL,
    --[[attack_multiplier]]attack_multiplier,
    --[[attack_order]]attack_order,
    --[[cant_move_over]]cant_move_over,
    --[[build_from]]UnitType.WORKSHOP,
    --[[build_skills]]build_skills
  )
  return ut
end

-- buildings
function rts_unit_factory.init_town_hall()
  local allowed_cmds = {CmdType.BUILD}
  local attack_multiplier = {}
  local attack_order = {}
  local cant_move_over = {Terrain.ROCK, Terrain.WATER}
  local build_skills = {
    __create_build_skill(UnitType.PEASANT, "p"),
  }
  local ut = __create_unit_template(
    --[[cost]]250,
    --[[hp]]350,
    --[[defense]]5,
    --[[speed]]0.0,
    --[[att]]0,
    --[[att_r]]0,
    --[[vis_r]]7,
    --[[cds]]{0, 0, 0, 50},
    --[[allowed_cmds]]allowed_cmds,
    --[[attr]]UnitAttr.ATTR_NORMAL,
    --[[attack_multiplier]]attack_multiplier,
    --[[attack_order]]attack_order,
    --[[cant_move_over]]cant_move_over,
    --[[build_from]]UnitType.PEASANT,
    --[[build_skills]]build_skills
  )
  return ut
end

function rts_unit_factory.init_barrack()
  local allowed_cmds = {CmdType.BUILD}
  local attack_multiplier = {}
  local attack_order = {}
  local cant_move_over = {Terrain.ROCK, Terrain.WATER}
  local build_skills = {
    __create_build_skill(UnitType.SPEARMAN, "e")
  }
  local ut = __create_unit_template(
    --[[cost]]150,
    --[[hp]]250,
    --[[defense]]5,
    --[[speed]]0.0,
    --[[att]]0,
    --[[att_r]]0,
    --[[vis_r]]7,
    --[[cds]]{0, 0, 0, 200},
    --[[allowed_cmds]]allowed_cmds,
    --[[attr]]UnitAttr.ATTR_NORMAL,
    --[[attack_multiplier]]attack_multiplier,
    --[[attack_order]]attack_order,
    --[[cant_move_over]]cant_move_over,
    --[[build_from]]UnitType.PEASANT,
    --[[build_skills]]build_skills
  )
  return ut
end

function rts_unit_factory.init_blacksmith()
  local allowed_cmds = {CmdType.BUILD}
  local attack_multiplier = {}
  local attack_order = {}
  local cant_move_over = {Terrain.ROCK, Terrain.WATER}
  local build_skills = {
    __create_build_skill(UnitType.SWORDMAN, "r")
  }
  local ut = __create_unit_template(
    --[[cost]]150,
    --[[hp]]250,
    --[[defense]]5,
    --[[speed]]0.0,
    --[[att]]0,
    --[[att_r]]0,
    --[[vis_r]]7,
    --[[cds]]{0, 0, 0, 200},
    --[[allowed_cmds]]allowed_cmds,
    --[[attr]]UnitAttr.ATTR_NORMAL,
    --[[attack_multiplier]]attack_multiplier,
    --[[attack_order]]attack_order,
    --[[cant_move_over]]cant_move_over,
    --[[build_from]]UnitType.PEASANT,
    --[[build_skills]]build_skills
  )
  return ut
end

function rts_unit_factory.init_stable()
  local allowed_cmds = {CmdType.BUILD}
  local attack_multiplier = {}
  local attack_order = {}
  local cant_move_over = {Terrain.ROCK, Terrain.WATER}
  local build_skills = {
    __create_build_skill(UnitType.CAVALRY, "c")
  }
  local ut = __create_unit_template(
    --[[cost]]150,
    --[[hp]]250,
    --[[defense]]5,
    --[[speed]]0.0,
    --[[att]]0,
    --[[att_r]]0,
    --[[vis_r]]7,
    --[[cds]]{0, 0, 0, 200},
    --[[allowed_cmds]]allowed_cmds,
    --[[attr]]UnitAttr.ATTR_NORMAL,
    --[[attack_multiplier]]attack_multiplier,
    --[[attack_order]]attack_order,
    --[[cant_move_over]]cant_move_over,
    --[[build_from]]UnitType.PEASANT,
    --[[build_skills]]build_skills
  )
  return ut
end

function rts_unit_factory.init_workshop()
  local allowed_cmds = {CmdType.BUILD}
  local attack_multiplier = {}
  local attack_order = {}
  local cant_move_over = {Terrain.ROCK, Terrain.WATER}
  local build_skills = {
    __create_build_skill(UnitType.CATAPULT, "u"),
    __create_build_skill(UnitType.DRAGON, "d"),
    __create_build_skill(UnitType.ARCHER, "r"),
  }
  local ut = __create_unit_template(
    --[[cost]]200,
    --[[hp]]250,
    --[[defense]]5,
    --[[speed]]0.0,
    --[[att]]0,
    --[[att_r]]0,
    --[[vis_r]]7,
    --[[cds]]{0, 0, 0, 200},
    --[[allowed_cmds]]allowed_cmds,
    --[[attr]]UnitAttr.ATTR_NORMAL,
    --[[attack_multiplier]]attack_multiplier,
    --[[attack_order]]attack_order,
    --[[cant_move_over]]cant_move_over,
    --[[build_from]]UnitType.PEASANT,
    --[[build_skills]]build_skills
  )
  return ut
end

function rts_unit_factory.init_aviary()
  local allowed_cmds = {CmdType.BUILD}
  local attack_multiplier = {}
  local attack_order = {}
  local cant_move_over = {Terrain.ROCK, Terrain.WATER}
  local build_skills = {
    -- __create_build_skill(UnitType.DRAGON, "d"),
  }
  local ut = __create_unit_template(
    --[[cost]]150,
    --[[hp]]250,
    --[[defense]]5,
    --[[speed]]0.0,
    --[[att]]0,
    --[[att_r]]0,
    --[[vis_r]]7,
    --[[cds]]{0, 0, 0, 200},
    --[[allowed_cmds]]allowed_cmds,
    --[[attr]]UnitAttr.ATTR_NORMAL,
    --[[attack_multiplier]]attack_multiplier,
    --[[attack_order]]attack_order,
    --[[cant_move_over]]cant_move_over,
    --[[build_from]]UnitType.PEASANT,
    --[[build_skills]]build_skills
  )
  return ut
end

function rts_unit_factory.init_archery()
  local allowed_cmds = {CmdType.BUILD}
  local attack_multiplier = {}
  local attack_order = {}
  local cant_move_over = {Terrain.ROCK, Terrain.WATER}
  local build_skills = {
    -- __create_build_skill(UnitType.ARCHER, "e"),
  }
  local ut = __create_unit_template(
    --[[cost]]150,
    --[[hp]]250,
    --[[defense]]5,
    --[[speed]]0.0,
    --[[att]]0,
    --[[att_r]]0,
    --[[vis_r]]7,
    --[[cds]]{0, 0, 0, 200},
    --[[allowed_cmds]]allowed_cmds,
    --[[attr]]UnitAttr.ATTR_NORMAL,
    --[[attack_multiplier]]attack_multiplier,
    --[[attack_order]]attack_order,
    --[[cant_move_over]]cant_move_over,
    --[[build_from]]UnitType.PEASANT,
    --[[build_skills]]build_skills
  )
  return ut
end

function rts_unit_factory.init_guard_tower()
  local allowed_cmds = {CmdType.ATTACK}
  local attack_multiplier = {
    {UnitType.PEASANT, 1.0},
    {UnitType.SWORDMAN, 1.0},
    {UnitType.SPEARMAN, 1.0},
    {UnitType.CAVALRY, 1.0},
    {UnitType.DRAGON, 1.0},
    {UnitType.ARCHER, 1.0},
    {UnitType.CATAPULT, 1.0},
    {UnitType.TOWN_HALL, 1.0},
    {UnitType.BARRACK, 1.0},
    {UnitType.STABLE, 1.0},
    {UnitType.BLACKSMITH, 1.0},
    {UnitType.WORKSHOP, 1.0},
    {UnitType.AVIARY, 1.0},
    {UnitType.ARCHERY, 1.0},
    {UnitType.GUARD_TOWER, 1.0},
  }
  -- attack order is the same as dragon
  local attack_order = {
    UnitType.DRAGON,
    UnitType.GUARD_TOWER,
    UnitType.ARCHER,
    UnitType.CATAPULT,
    UnitType.CAVALRY,
    UnitType.SWORDMAN,
    UnitType.SPEARMAN,
    UnitType.PEASANT,
    UnitType.TOWN_HALL
  }
  local cant_move_over = {Terrain.ROCK, Terrain.WATER}
  local build_skills = {}
  local ut = __create_unit_template(
    --[[cost]]150,
    --[[hp]]100,
    --[[defense]]3,
    --[[speed]]0.0,
    --[[att]]10,
    --[[att_r]]5,
    --[[vis_r]]7,
    --[[cds]]{0, 30, 0, 0},
    --[[allowed_cmds]]allowed_cmds,
    --[[attr]]UnitAttr.ATTR_NORMAL,
    --[[attack_multiplier]]attack_multiplier,
    --[[attack_order]]attack_order,
    --[[cant_move_over]]cant_move_over,
    --[[build_from]]UnitType.PEASANT,
    --[[build_skills]]build_skills
  )
  return ut
end
