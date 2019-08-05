// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include <list>
#include <cmath>

#include "engine/game_state_ext.h"
#include "engine/game_action.h"
#include "ai/ai_option.h"

inline int computeUniqueId(int threadId, int playerId) {
  assert(playerId >= 0 && playerId <= 16); // prevent overflow?
  return ((playerId << 20) + threadId);
}

class AI {
 public:
  AI(const AIOption& opt, int threadId)
      : maxBuildQueueSize(1),
        option_(opt),
        threadId_(threadId) {}

  AI(const AI&) = delete;
  AI& operator=(const AI&) = delete;

  virtual ~AI() {}

  virtual void setId(int id) {
    id_ = id;
    uniqueId_ = computeUniqueId(threadId_, getId());
    rng.seed(uniqueId_);
  }

  int getId() const {
    return id_;
  }

  // Given the current state, perform action and send the action to _a;
  // Return false if this procedure fails.
  virtual bool act(const RTSStateExtend&, RTSAction*) {
    return true;
  }

  virtual bool newGame(const RTSStateExtend&) {
    return true;
  }

  virtual bool endGame(const RTSStateExtend&) {
    return true;
  }

  bool respectFow() const {
    return option_.fow;
  }

  float resourceScale() const {
    float scale = option_.resource_scale;
    scale = round(scale * 10) / 10.0;
    return scale;
  }

  virtual int frameSkip() const {
    return option_.fs;
  }

  int threadId() const {
    return threadId_;
  }

  std::list<UnitType> buildQueue;
  const size_t maxBuildQueueSize;
  std::mt19937 rng;

 protected:
  AIOption option_;
  int threadId_;
  int id_; // id in a game (thread)
  int uniqueId_; // this id is unique across all ai in all games
};
