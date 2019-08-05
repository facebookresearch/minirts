#pragma once

#include "data_block.h"

namespace tube {

class EpisodicTrajectory {
 public:
  EpisodicTrajectory(const std::string& name,
                     int blockLen,
                     const std::vector<int64_t>& sizes,
                     torch::ScalarType dtype)
      : name(name)
      , blockLen(blockLen)
      , dtype(dtype)
      , sizes(sizes)
      , buffer(std::make_shared<DataBlock>(
            name, utils::pushLeft(blockLen, sizes), dtype)) {
  }

  int pushBack(torch::Tensor&& t) {
    assert(t.dtype() == dtype);
    assert(t.sizes() == sizes);
    trajectory_.push_back(t);
    return (int)trajectory_.size();
  }

  bool prepareForSend() {
    if ((int)trajectory_.size() < blockLen) {
      return false;
    }
    for (int i = 0; i < blockLen; ++i) {
      buffer->data[i].copy_(trajectory_.front());
      trajectory_.pop_front();
    }
    return true;
  }

  int len() {
    return (int)trajectory_.size();
  }

  const std::string name;
  const int blockLen;
  const torch::ScalarType dtype;
  const std::vector<int64_t> sizes;

  std::shared_ptr<DataBlock> buffer;

 private:
  std::deque<torch::Tensor> trajectory_;
};
}
