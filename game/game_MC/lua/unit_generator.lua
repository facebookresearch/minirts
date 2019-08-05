-- Copyright (c) Facebook, Inc. and its affiliates.
-- All rights reserved.
--
-- This source code is licensed under the license found in the
-- LICENSE file in the root directory of this source tree.
--
local random = require 'random'

rts_unit_generator = {}

function __make_p(x, y)
  local p = PointF.new()
  p:set_int_xy(x, y)
  return p
end

local unit_type_to_cell = {}
unit_type_to_cell[UnitType.RESOURCE] = "X"
unit_type_to_cell[UnitType.PEASANT] = "W"
unit_type_to_cell[UnitType.SWORDMAN] = "S"
unit_type_to_cell[UnitType.SPEARMAN] = "R"
unit_type_to_cell[UnitType.CAVALRY] = "T"
unit_type_to_cell[UnitType.ARCHER] = "C"
unit_type_to_cell[UnitType.DRAGON] = "F"
unit_type_to_cell[UnitType.CATAPULT] = "U"
unit_type_to_cell[UnitType.BARRACK] = "A"
unit_type_to_cell[UnitType.BLACKSMITH] = "O"
unit_type_to_cell[UnitType.STABLE] = "H"
unit_type_to_cell[UnitType.WORKSHOP] = "K"
unit_type_to_cell[UnitType.AVIARY] = "V"
unit_type_to_cell[UnitType.ARCHERY] = "Y"
unit_type_to_cell[UnitType.GUARD_TOWER] = "D"
unit_type_to_cell[UnitType.TOWN_HALL] = "B"

function can_pass(proxy, x, y, with_margin)
  local max_x = proxy:get_x_size()
  local map_y = proxy:get_y_size()

  function verify(x, y)
    if not (x >= 0 and x < max_x and y >= 0 and y < map_y) then
      return false
    end
    local ty = proxy:get_slot_type(x, y, 0)
    return not (ty == Terrain.WATER or ty == Terrain.ROCK)
  end

  if not verify(x, y) then
    return false
  end

  if with_margin then
    local dx = {-1, 1, 0, 0}
    local dy = {0, 0, -1, 1}
    for i = 1, #dx do
      if not verify(x + dx[i], y + dy[i]) then
        return false
      end
    end
  end
  return true
end

function can_reach(proxy, sx, sy, tx, ty)
  local max_x = proxy:get_x_size()
  local map_y = proxy:get_y_size()
  local dx = {-1, 1, 0, 0}
  local dy = {0, 0, -1, 1}

  function dfs(x, y, tx, ty, visited)
    if x == tx and y == ty then
      return true
    end
    visited[x * map_y + y] = true
    for i = 1, #dx do
      local nx = x + dx[i]
      local ny = y + dy[i]
      if (nx >= 0 and nx < max_x and ny >= 0 and ny < map_y
          and visited[nx * map_y + ny] == nil) then
        if can_pass(proxy, nx, ny, false) and dfs(nx, ny, tx, ty, visited) then
          return true
        end
      end
    end
    return false
  end

  return dfs(sx, sy, tx, ty, {})
end

function get_manhattan_circle(proxy, cx, cy, r, circle)
  local max_x = proxy:get_x_size()
  local max_y = proxy:get_y_size()

  for xx = math.max(0, cx - r), math.min(max_x - 1, cx + r) do
    dx = math.abs(xx - cx)
    local ylen = r - dx
    y1 = cy - ylen
    y2 = cy + ylen
    if can_pass(proxy, xx, y1, true) and can_reach(proxy, xx, y1, cx, cy) then
      table.insert(circle, {xx, y1})
    end
    if (y2 ~= y1 and can_pass(proxy, xx, y2, true) and
        can_reach(proxy, xx, y2, cx, cy)) then
      table.insert(circle, {xx, y2})
    end
  end
end

function get_manhattan_ring(proxy, cx, cy, r_min, r_max, ring)
  -- print('manhattan ring: ', r_min, ' -> ', r_max)
  local max_x = proxy:get_x_size()
  local max_y = proxy:get_y_size()

  for dx = -r_max, r_max do
    for dy = -r_max, r_max do
      local d = math.abs(dx) + math.abs(dy)
      if d >= r_min and d <= r_max then
        local xx = cx + dx
        local yy = cy + dy
        if can_pass(proxy, xx, yy, true) and can_reach(proxy, xx, yy, cx, cy) then
          table.insert(ring, {xx, yy})
        end
      end
    end
  end
end

function add_unit_and_clear_region(
    proxy, unit_type, x, y, radius, player_id, taken)
  local max_x = proxy:get_x_size()
  local map_y = proxy:get_y_size()
  for i = math.max(x - radius, 0), math.min(max_x - 1, x + radius) do
    for j = math.max(y - radius, 0), math.min(map_y - 1, y + radius) do
      proxy:set_slot_type(Terrain.GRASS, i, j, 0)
    end
  end
  proxy:send_cmd_create(unit_type, __make_p(x, y), player_id, 0)
  taken[x * map_y + y] = true
end

function add_resources_on_radius(
    proxy, num_resources, cx, cy, radius, fair, player_id, rng, taken)
  local max_x = proxy:get_x_size()
  local map_y = proxy:get_y_size()
  local r_min = 1

  for i = 1, num_resources do
    local r = i * radius
    local valid_places = {}
    if fair then
      get_manhattan_circle(proxy, cx, cy, r, valid_places)
    else
      get_manhattan_ring(proxy, cx, cy, r_min, r, valid_places)
      r_min = r
    end
    if #valid_places == 0 then
      return
    end
    local rand_idx = rng(1, #valid_places)
    local coord = valid_places[rand_idx]
    local x = coord[1]
    local y = coord[2]
    -- print('dist: ', math.abs(x - cx) + math.abs(y - cy))
    proxy:send_cmd_create(UnitType.RESOURCE, __make_p(x, y), player_id, 0)
    taken[x * map_y + y] = true
  end
end

function add_units_within_region(
    proxy, num_units, unit_type, x, y, radius, player_id, rng, taken)
  local max_x = proxy:get_x_size()
  local map_y = proxy:get_y_size()
  for i = 1, num_units do
    local found = false
    local iter = 0
    while iter <= 20 do
      iter = iter + 1
      local xx = x + rng(-radius, radius)
      local yy = y + rng(-radius, radius)
      if can_pass(proxy, xx, yy, true) and taken[xx * map_y + yy] == nil then
        proxy:send_cmd_create(unit_type, __make_p(xx, yy), player_id, 0)
        taken[xx * map_y + yy] = true
        break
      end
    end
  end
end

function add_base_resource_and_workers(
    proxy,
    x,
    y,
    radius,
    num_resources,
    player_id,
    rng,
    fair,
    num_peasants,
    num_extra_units,
    taken)
  -- town hall and clear regions
  add_unit_and_clear_region(
    proxy, UnitType.TOWN_HALL, x, y, radius, player_id, taken)
  -- resource
  add_resources_on_radius(
    proxy, num_resources, x, y, radius, fair, player_id, rng, taken)
  -- add resource in neutral location
  for i = 1, 3 do
    add_resources_on_radius(
      proxy, 1, x, y, proxy:get_x_size() // 2, true, player_id, rng, taken)
  end
  -- peasant
  add_units_within_region(
    proxy, num_peasants, UnitType.PEASANT, x, y, 5, player_id, rng, taken)

  -- -- generate free units at the beginning
  -- local free_types = {UnitType.SPEARMAN, UnitType.SWORDMAN, UnitType.CAVALRY}
  -- local rand_idx = rng(1, #free_types)
  -- -- add_units(1, free_types[rand_idx], proxy, x, y, player_id, rng, taken)

  for i = 1, num_extra_units do
    local pass = rng(1, 2)
    if pass == 2 then
      -- print('player:', player_id, ' create extra unit')
      local candidate_types
      if player_id == 0 then
        candidate_types = {
          UnitType.SPEARMAN,
          UnitType.SWORDMAN,
          UnitType.CAVALRY,
          UnitType.ARCHER,
          UnitType.DRAGON,
          UnitType.CATAPULT,
          UnitType.GUARD_TOWER,
          UnitType.WORKSHOP,
          UnitType.BLACKSMITH,
          UnitType.STABLE,
          UnitType.BARRACK
        }
      else
        candidate_types = {
          UnitType.SPEARMAN,
          UnitType.SWORDMAN,
          UnitType.CAVALRY,
          UnitType.ARCHER,
          UnitType.DRAGON,
          UnitType.CATAPULT,
          UnitType.GUARD_TOWER,
        }
      end
      local rand_idx = rng(1, #candidate_types)
      add_units_within_region(
        proxy, 1, candidate_types[rand_idx], x, y, 5, player_id, rng, taken)
    end
  end
end

function generate_bases(
    proxy,
    rng,
    player_ids,
    resource_dist,
    num_resources,
    fair,
    num_peasants,
    num_extra_units)
  local max_x = proxy:get_x_size()
  local map_y = proxy:get_y_size()
  assert(max_x == map_y)
  local lb = 2 -- math.ceil(map_y / 9)
  local ub = 4 -- math.floor(map_y / 7)

  local rotate = (rng(0, 999) / 1000) < 0.5
  local swap = (rng(0, 999) / 1000) < 0.5
  local basexy = {}
  local found = false
  local iter = 0
  local taken = {}
  while iter < 100 do
    iter = iter + 1
    local x1 = rng(lb, max_x - 1 - lb)
    local x2 = rng(lb, max_x - 1 - lb)
    local y1 = rng(lb, ub)
    local y2 = map_y - 1 - rng(lb, ub)
    -- swap the coordinates
    if rotate then
      x1, y1 = y1, x1
      x2, y2 = y2, x2
    end
    if swap then
      x1, x2 = x2, x1
      y1, y2 = y2, y1
    end
    if can_reach(proxy, x1, y1, x2, y2) then
      if resource_dist == 0 then
        resource_dist = rng(4, 6)
      end
      add_base_resource_and_workers(
        proxy,
        x1,
        y1,
        resource_dist,
        num_resources,
        player_ids[1],
        rng,
        fair,
        num_peasants,
        num_extra_units,
        taken)
      add_base_resource_and_workers(
        proxy,
        x2,
        y2,
        resource_dist,
        num_resources,
        player_ids[2],
        rng,
        fair,
        num_peasants,
        num_extra_units,
        taken)
      basexy[1] = {x1, y1}
      basexy[2] = {x2, y2}
      break
    end
  end
  if iter == 100 then
    print("WARNING: fail to generate map")
  end

  local block_size_x = math.floor(max_x / 3)
  local block_size_y = math.floor(map_y / 3)
  local block_taken = {}
  for i = 1, #taken do
    if taken[i] then
      local x = i / map_y
      local y = i % map_y
      block_taken[math.floor(x / block_size_x) * 3 + math.floor(y / block_size_y)] = true
    end
  end
end

function rts_unit_generator.generate_random(
    proxy,
    num_players,
    seed,
    resource,
    resource_dist,
    num_resources,
    fair,
    num_peasants,
    num_extra_units)
  local rng = random.get_rng(seed)
  -- provide with correct player ids
  local player_ids = {0, 1}
  if num_players == 3 then
    player_ids = {0, 2}
  end
  generate_bases(
    proxy,
    rng,
    player_ids,
    resource_dist,
    num_resources,
    fair,
    num_peasants,
    num_extra_units)

  -- change resources
  for pid = 1, #player_ids do
    local player_id = player_ids[pid]
    proxy:send_cmd_change_player_resource(player_id, resource)
  end
end
