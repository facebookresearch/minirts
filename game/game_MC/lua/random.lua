-- Copyright (c) Facebook, Inc. and its affiliates.
-- All rights reserved.
--
-- This source code is licensed under the license found in the
-- LICENSE file in the root directory of this source tree.
--
local random = {}

-- generate random number from [min, max] (both inclusive)
function random.get_rng(seed)
  local __seed = seed

  function fast_random(min, max)
    __seed = (__seed * 1009 + 9007) % 65537
    num = (__seed % (max - min + 1)) + min
    return num
  end

  return fast_random
end

return random
