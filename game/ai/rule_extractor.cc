// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#include "rule_extractor.h"

bool RuleExtractor::skim(const RTSStateExtend& state, const AI& ai) {
  if (!useMovingAvg) {
    return false;
  }

  Preload preload;
  preload.GatherInfo(
      state.env(),
      ai.getId(),
      state.receiver(),
      ai.buildQueue,
      ai.respectFow());

  const auto& enemyTroops = preload.EnemyTroops();
  auto accessor = movingEnemyCount_.getBuffer().accessor<float, 1>();
  assert(accessor.size(0) == (int)enemyTroops.size());
  for (int i = 0; i < (int)enemyTroops.size(); ++i) {
    int count = enemyTroops[i].size();
    accessor[i] *= movingAvgDecay;
    accessor[i] += count * (1 - movingAvgDecay);
  }
  return true;
}

void RuleExtractor::computeFeatures(
    const Preload& preload,
    const CmdReceiver&,
    const GameEnv& env,
    PlayerId playerId,
    bool) {
  computeUnitCount(preload);
  computeUnitConsCount(preload);
  // movingEnemyCount is set in skim
  computeResourceBin(preload.Resource());

  if (env.GetTermination()) {
    terminal_.getBuffer()[0] = (float)true;
    auto winner = env.GetWinnerId();
    reward_.getBuffer()[0] = (winner == playerId ? 1.0 : -1.0);
  } else {
    terminal_.getBuffer()[0] = (float)false;
    reward_.getBuffer()[0] = 0;
  }
}

void RuleExtractor::newGame() {
  prevAction_.getBuffer().zero_();
  movingEnemyCount_.getBuffer().zero_();
}

int RuleExtractor::pushGameFeature() {
  int idx = unitCount_.pushBufferToTrajectory();
  assert(idx == unitConsCount_.pushBufferToTrajectory());
  assert(idx == movingEnemyCount_.pushBufferToTrajectory());
  assert(idx == resourceBin_.pushBufferToTrajectory());
  assert(idx == prevAction_.pushBufferToTrajectory());
  return idx;
}

int RuleExtractor::pushLastRAndTerminal() {
  int idx = terminal_.pushBufferToTrajectory();
  assert(idx == reward_.pushBufferToTrajectory());
  return idx;
}

int RuleExtractor::pushActionAndPolicy() {
  int idx = action_.pushBufferToTrajectory();
  assert(idx == policy_.pushBufferToTrajectory());
  return idx;
}

void RuleExtractor::postActUpdate() {
  prevAction_.getBuffer().copy_(action_.getBuffer());
}

// private functions
void RuleExtractor::computeUnitCount(const Preload& preload) {
  const auto& myTroops = preload.MyTroops();
  const auto& enemyTroops = preload.EnemyTroops();
  auto accessor = unitCount_.getBuffer().accessor<float, 1>();
  assert(accessor.size(0) == (int)(myTroops.size() + enemyTroops.size()));
  for (int i = 0; i < (int)myTroops.size(); ++i) {
    accessor[i] = (float)myTroops[i].size();
  }
  int offset = (int)myTroops.size();
  for (int i = 0; i < (int)enemyTroops.size(); ++i) {
    accessor[offset + i] = (float)enemyTroops[i].size();
  }
}

void RuleExtractor::computeUnitConsCount(const Preload& preload) {
  const std::vector<int>& counters = preload.CntUnderConstruction();
  auto accessor = unitConsCount_.getBuffer().accessor<float, 1>();
  assert(accessor.size(0) == (int)counters.size());
  for (int i = 0; i < (int)counters.size(); ++i) {
    accessor[i] = (float)counters[i];
  }
}

void RuleExtractor::computeResourceBin(int resource) {
  int binIdx = resource / resourceBinSize;
  if (binIdx >= numResourceBin) {
    binIdx = numResourceBin - 1;
  }
  auto accessor = resourceBin_.getBuffer().accessor<float, 1>();
  assert(numResourceBin == accessor.size(0));
  for (int i = 0; i < numResourceBin; ++i) {
    if (binIdx == i) {
      accessor[i] = 1;
    } else {
      accessor[i] = 0;
    }
  }
}
