// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include <vector>

#include "data_block.h"

using tube::DataBlock;
using tube::FixedLengthTrajectory;

class OnehotInstruction {
 public:
  OnehotInstruction(int numInstruction, int padInstruction, int tLen, int maxRawChar)
      : numInstruction(numInstruction),
        padInstruction(padInstruction),
        maxRawChar(maxRawChar),
        framePassed_("frame_passed", tLen + 1, {1}, torch::kInt64),
        prevInst_("prev_inst", tLen + 1, {1}, torch::kInt64),
        cont_("cont", tLen, {1}, torch::kInt64),
        contPi_("cont_pi", tLen, {2}, torch::kFloat32),
        inst_("inst", tLen, {1}, torch::kInt64),
        instPi_("inst_pi", tLen, {numInstruction}, torch::kFloat32)
  {
    rawInst_ = std::make_shared<DataBlock>(
        "raw_inst", std::initializer_list<int64_t>{maxRawChar}, torch::kInt64);

    // extra feature for executor
    histInst_ = std::make_shared<DataBlock>(
        "hist_inst", std::initializer_list<int64_t>{histLen_}, torch::kInt64);
    histInstDiff_ = std::make_shared<DataBlock>(
        "hist_inst_diff", std::initializer_list<int64_t>{histLen_}, torch::kInt64);

    // need to set here for the first game
    prevInst_.getBuffer()[0] = padInstruction;
  }

  void newGame() {
    framePassed_.getBuffer()[0] = 0;
    prevInst_.getBuffer()[0] = padInstruction;

    step_ = 0;
    allInst_.clear();
    allInstStep_.clear();
    for (int i = 0; i < histLen_; ++i) {
      allInst_.push_back(padInstruction);
      allInstStep_.push_back(0);
    }
    extractHistInst();
  }

  bool sameInstruction() const {
    // TODO: BE AWARE
    return cont_.getBuffer().item<int64_t>() > 0;
  }

  std::vector<int64_t> getRawInstruction() const {
    std::vector<int64_t> inst(maxRawChar);
    auto accessor = rawInst_->getBuffer().accessor<int64_t, 1>();
    assert(accessor.size(0) == maxRawChar);
    for (int i = 0; i < maxRawChar; ++i) {
      inst[i] = accessor[i];
    }
    return inst;
  }

  std::vector<std::shared_ptr<DataBlock>> getTrainSend() const {
    std::vector<std::shared_ptr<DataBlock>> blocks = {
      framePassed_.trajectory,
      prevInst_.trajectory,
      cont_.trajectory,
      contPi_.trajectory,
      inst_.trajectory,
      instPi_.trajectory
    };
    return blocks;
  }

  std::vector<std::shared_ptr<DataBlock>> getActSend() const {
    return {framePassed_.buffer, prevInst_.buffer, histInst_, histInstDiff_};
  }

  std::vector<std::shared_ptr<DataBlock>> getActReply() const {
    return {cont_.buffer, contPi_.buffer, inst_.buffer, instPi_.buffer, rawInst_};
  }

  void pushGameFeature() {
    framePassed_.pushBufferToTrajectory();
    prevInst_.pushBufferToTrajectory();
  }

  void pushActionAndPolicy() {
    cont_.pushBufferToTrajectory();
    contPi_.pushBufferToTrajectory();
    inst_.pushBufferToTrajectory();
    instPi_.pushBufferToTrajectory();
  }

  void postActUpdate() {
    if (sameInstruction()) {
      framePassed_.getBuffer()[0] += 1;
      // return;
    } else {
      // refresh prevInst & frame passed
      prevInst_.getBuffer().copy_(inst_.getBuffer());
      framePassed_.getBuffer()[0] = 1;

      // update inst history
      allInst_.push_back(inst_.getBuffer().item<int64_t>());
      allInstStep_.push_back(step_);
    }
    step_ += 1;
    // extract feat for next step, this must come after step_ += 1
    // TOOD: should be put into computeFeature function
    extractHistInst();
  }

  const int numInstruction;
  const int padInstruction;
  const int maxRawChar;

 protected:
  void extractHistInst() {
    auto histInst = histInst_->getBuffer().accessor<int64_t, 1>();
    auto histInstDiff = histInstDiff_->getBuffer().accessor<int64_t, 1>();
    assert(allInst_.size() == allInstStep_.size());
    assert((int)allInst_.size() >= histLen_);
    int offset = allInst_.size() - histLen_;
    for (int i = 0; i < histLen_; ++i) {
      histInst[i] = allInst_[i + offset];
      histInstDiff[i] = (step_ - allInstStep_[i + offset]) / 5;
    }
  }

  FixedLengthTrajectory framePassed_;
  FixedLengthTrajectory prevInst_;
  FixedLengthTrajectory cont_;
  FixedLengthTrajectory contPi_;
  FixedLengthTrajectory inst_;
  FixedLengthTrajectory instPi_;

  std::shared_ptr<DataBlock> histInst_;
  std::shared_ptr<DataBlock> histInstDiff_;

  int step_ = 0;
  std::vector<int64_t> allInst_;
  std::vector<int64_t> allInstStep_;
  // TODO: this should be passed from python
  int histLen_ = 5;
  std::shared_ptr<DataBlock> rawInst_;
};
