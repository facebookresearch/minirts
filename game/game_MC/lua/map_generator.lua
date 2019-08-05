-- Copyright (c) Facebook, Inc. and its affiliates.
-- All rights reserved.
--
-- This source code is licensed under the license found in the
-- LICENSE file in the root directory of this source tree.
--
local random = require 'random'

rts_map_generator = {}
margin = 5

function rts_map_generator.generate_terrain(map, location_func, ty, rng)
  local xs, ys = location_func()
  for i = 1, #xs do
    map:set_slot_type(ty, xs[i] - 1, ys[i] - 1, 0)
  end
end

function rts_map_generator.generate_base(map)
  local x = rng(margin, map:get_x_size() - margin)
  local y = rng(margin, map:get_y_size() - margin)
  return x, y
end

function is_in(map, x, y)
  local X = map:get_x_size()
  local Y = map:get_y_size()
  if x >= 0 and x < X and y >= 0 and y < Y then
    return true
  else
    return false
  end
end

function sign(x)
  if x < 0 then
    return -1
  elseif x > 1 then
    return 1
  else
    return 0
  end
end

function add_line(map, sx, sy, tx, ty, terrain, has_hole, rng)
  local X = map:get_x_size()
  local Y = map:get_y_size()
  local dx = math.max(1, math.abs(sx - tx))
  local dy = math.max(1, math.abs(sy - ty))
  local d = math.max(dx, dy)
  local sigx = sign(tx - sx)
  local sigy = sign(ty - sy)
  local hole_size = rng(2, 4)
  local hole_loc = rng(0, d - hole_size)

  for i = 0, d do
    local x = sigx * math.floor(i / d * dx) + sx
    local y = sigy * math.floor(i / d * dy) + sy
    local put_hole = has_hole and i >= hole_loc and i - hole_loc < hole_size
    if not put_hole then
      map:set_slot_type(terrain, x, y, 0)
      -- add margin
      if dy > dx then
        map:set_slot_type(terrain, x - 1, y, 0)
        --map:set_slot_type(terrain, x + 1, y, 0)
      else
        map:set_slot_type(terrain, x, y - 1, 0)
        --map:set_slot_type(terrain, x, y + 1, 0)
      end
    end
  end
end

function gen_holes_mask(N, rng)
  local n_holes = rng(2, math.min(3, N))
  local holes_mask = {}
  for i = 1, N do
    holes_mask[i] = false
  end
  for i = 1, n_holes do
    local found = false
    while not found do
      local x = rng(1, N)
      if holes_mask[x] == false then
        holes_mask[x] = true
        found = true
      end
    end
  end
  return holes_mask
end

function rts_map_generator.generate_water_hor(map, rng)
  local N = 3
  local delta = math.floor(map:get_x_size() / N)
  local last_y = rng(1, N - 1)
  local holes_mask = gen_holes_mask(N, rng)
  for xi = 1, N do
    local ly = math.max(last_y - 2, 1)
    local ry = math.min(last_y + 2, N - 1)
    local y = rng(ly, ry)
    add_line(
      map,
      (xi - 1) * delta,
      last_y * delta,
      xi * delta,
      y * delta,
      Terrain.WATER,
      holes_mask[xi],
      rng)
    last_y = y
  end
end

function rts_map_generator.generate_water_ver(map, rng)
  local N = 3
  local delta = math.floor(map:get_x_size() / N)
  local last_x = rng(1, N - 1)
  local holes_mask = gen_holes_mask(N, rng)
  for yi = 1, N do
    local lx = math.max(last_x - 2, 1)
    local rx = math.min(last_x + 2, N - 1)
    local x = rng(lx, rx)
    add_line(
      map,
      last_x * delta,
      (yi - 1) * delta,
      x * delta, yi * delta,
      Terrain.WATER,
      holes_mask[yi],
      rng)
    last_x = x
  end
end

function rts_map_generator.generate_water(map, rng)
  rts_map_generator.generate_water_hor(map, rng)
  rts_map_generator.generate_water_ver(map, rng)
end

function rts_map_generator.generate_soil_and_sand(map, rng)
  local X = map:get_x_size()
  local Y = map:get_y_size()
  local dx = {-1, -1, 1, 1}
  local dy = {-1, 1, -1, 1}
  for x = 0, X - 1 do
    for y = 0, Y -1 do
      local any = false
      if map:get_slot_type(x, y, 0) == Terrain.GRASS then
        for i=1, #dx do
          local nx = x + dx[i]
          local ny = y + dy[i]
          if (nx >= 0 and nx < X and ny >= 0 and ny < Y and
              map:get_slot_type(nx, ny, 0) == Terrain.WATER) then
            local p = rng(0, 999) / 1000
            if p < 0.1 then
              map:set_slot_type(Terrain.SOIL, x, y, 0)
              any = true
            elseif p < 0.3 then
              map:set_slot_type(Terrain.SAND, x, y, 0)
              any = true
            end
            break
          end
        end
      end
      if not any then
          local p = rng(0, 999) / 1000
          if p < 0.01 then
              map:set_slot_type(Terrain.SAND, x, y, 0)
          end
      end
    end
  end
end

function rts_map_generator.generate_rock(map, rng)
  local X = map:get_x_size()
  local Y = map:get_y_size()
  for x = 0, X - 1 do
    for y = 0, Y - 1 do
      if map:get_slot_type(x, y, 0) == Terrain.GRASS then
        local p = rng(0, 999) / 1000
        if p < 0.005 then
          map:set_slot_type(Terrain.ROCK, x, y, 0)
        end
      end
    end
  end
end

function rts_map_generator.generate_random(map, num_players, seed, no_terrain)
  local rng = random.get_rng(seed)
  map:clear_map()
  local X = 32
  local Y = 32

  map:init_map(X, Y, 1)
  if not no_terrain then
    rts_map_generator.generate_water(map, rng)
    -- rts_map_generator.generate_rock(map, rng)
    -- rts_map_generator.generate_soil_and_sand(map)
  end

  map:reset_intermediates()
end
