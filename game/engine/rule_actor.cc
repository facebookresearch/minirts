// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#include <list>

#include "gamedef.h"
#include "rule_actor.h"
#include "utils.h"

bool RuleActor::store_cmd(const Unit* u, CmdBPtr&& cmd, AssignedCmds* m) const {
  assert(cmd.get() != nullptr);

  // at most one cmd per tick
  if (UnitHasCmd(u->GetId(), *m)) {
    return false;
  }

  (*m)[u->GetId()] = std::move(cmd);
  return true;
}

// void RuleActor::batch_store_cmds(
//     const std::vector<const Unit*>& subset,
//     const CmdBPtr& cmd,
//     bool preemptive,
//     AssignedCmds* m) const {
//   for (const Unit* u : subset) {
//     const CmdDurative* curr_cmd = _receiver.GetUnitDurativeCmd(u->GetId());
//     if (curr_cmd == nullptr || preemptive) {
//       store_cmd(u, cmd->clone(), m);
//     }
//   }
// }

bool RuleActor::GatherInfo(const GameEnv& env,
                           const std::list<UnitType>& build_queue,
                           bool respect_fow) {
  if (!_copied_preload) {
    _preload.GatherInfo(env, _player_id, _receiver, build_queue, respect_fow);
  }
  return _preload.Ok();
}
