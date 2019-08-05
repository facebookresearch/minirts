// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#include "cmd.h"
#include "cmd.gen.h"
#include "cmd_receiver.h"
#include "common.h"
#include "game_env.h"
#include "gamedef.h"
#include "utils.h"

SERIALIZER_ANCHOR_INIT(CmdBase);
SERIALIZER_ANCHOR_INIT(CmdImmediate);
SERIALIZER_ANCHOR_INIT(CmdDurative);
SERIALIZER_ANCHOR_INIT(CmdControl);

std::map<std::string, int> CmdTypeLookup::_name2idx;
std::map<int, std::string> CmdTypeLookup::_idx2name;
std::string CmdTypeLookup::_null;
std::mutex CmdTypeLookup::_mutex;

bool CmdDurative::Run(const GameEnv& env, CmdReceiver* receiver) {
  assert(env.IsGameActive());
  // If we run this command for the first time, register it in the receiver.
  if (just_started()) {
    receiver->StartDurativeCmd(env, this);
  }

  // Run the script.
  bool ret = run(env, receiver);

  _tick = receiver->GetNextTick();
  return ret;
}

bool CmdDurative::Cleanup(const GameEnv& env, CmdReceiver* receiver) {
  return cleanup(env, receiver);
}

// ============= Commands ==============
// ----- Move
bool CmdTacticalMove::run(GameEnv* env, CmdReceiver*) {
  assert(env->IsGameActive());

  Unit* u = env->GetUnit(_id);
  if (u == nullptr) {
    return false;
  }
  RTSMap& m = env->GetMap();

  // Move a unit.
  if (m.MoveUnit(_id, _p)) {
    // if (_p.x < 0 || _p.y < 0) {
    //   std::cout << "tac move: " << _p.x << ", " <<  _p.y << std::endl;
    // }
    u->SetPointF(_p);
    u->GetProperty().CD(CD_MOVE).Start(_tick);
    return true;
  } else {
    return false;
  }
}

bool CmdCDStart::run(GameEnv* env, CmdReceiver*) {
  assert(env->IsGameActive());

  Unit* u = env->GetUnit(_id);
  if (u == nullptr) {
    return false;
  }
  u->GetProperty().CD(_cd_type).Start(_tick);
  return true;
}

bool CmdEmitBullet::run(GameEnv* env, CmdReceiver*) {
  assert(env->IsGameActive());

  if (_id == INVALID) {
    return false;
  }
  Unit* u = env->GetUnit(_id);
  if (u == nullptr) {
    return false;
  }
  // cout << "Bullet: " << micro_cmd.PrintInfo() << endl;
  Bullet b(_p, _id, _att, _speed);
  b.SetTargetUnitId(_target);
  env->AddBullet(b);
  return true;
}

bool CmdCreate::run(GameEnv* env, CmdReceiver* receiver) {
  assert(env->IsGameActive());
  if (GameDef::IsUnitTypeBuilding(_build_type)) {
    RemoveTemporaryBuilding(_p, _build_type, _player_id, env, receiver);
  }

  // Create a unit at a location
  if (!env->AddUnit(_tick, _build_type, _p, _player_id, _expiration)) {
    // If failed, money back!
    env->GetPlayer(_player_id).ChangeResource(_resource_used);
    return false;
  }
  return true;
}

bool CmdRemove::run(GameEnv* env, CmdReceiver* receiver) {
  assert(env->IsGameActive());
  // run cleanup before we are removing unit
  receiver->CleanupDurativeCmd(*env, _id);
  if (env->RemoveUnit(_id)) {
    receiver->FinishDurativeCmd(_id);
    return true;
  } else {
    return false;
  }
}

bool CmdLoadMap::run(GameEnv* env, CmdReceiver*) {
  serializer::loader loader(false);
  if (!loader.read_from_file(_s)) {
    return false;
  }
  loader >> env->GetMap();
  return true;
}

bool CmdSaveMap::run(GameEnv* env, CmdReceiver*) {
  serializer::saver saver(false);
  saver << env->GetMap();
  if (!saver.write_to_file(_s)) {
    return false;
  }
  return true;
}

bool CmdRandomSeed::run(GameEnv* env, CmdReceiver*) {
  assert(env->IsGameActive());
  env->SetSeed(_seed);
  return true;
}

bool CmdComment::run(GameEnv* env, CmdReceiver*) {
  assert(env->IsGameActive());
  // COMMENT command does not do anything, but leave a record.
  return true;
}

bool CmdFreezeGame::run(GameEnv* env, CmdReceiver*) {
  if (_freeze) {
    env->ChangeGameStatus(FROZEN_STATUS);
  } else {
    env->ChangeGameStatus(ACTIVE_STATUS);
  }
  return true;
}
