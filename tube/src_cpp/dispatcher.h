#pragma once

#include "data_block.h"
#include "data_channel.h"

namespace tube {

class Dispatcher {
 public:
  Dispatcher(std::shared_ptr<DataChannel> dc)
      : dc_(std::move(dc)) {
  }

  void addDataBlocks(const std::vector<std::shared_ptr<DataBlock>>& send,
                     const std::vector<std::shared_ptr<DataBlock>>& reply) {
    for (auto b : send) {
      auto ret = sendTensors_.insert({b->name, b->data});
      if (!ret.second) {
        std::cout << "Error: duplicated sendkey for dispatcher, "
                  << "key=" << b->name << ", DataChannel=" << dc_->name;
        assert(false);
      }
    }

    for (auto b : reply) {
      auto ret = replyTensors_.insert({b->name, b->data});
      if (!ret.second) {
        std::cout << "Error: duplicated replykey for dispatcher, "
                  << "key=" << b->name << ", DataChannel=" << dc_->name;
        assert(false);
      }
    }
    dc_->createOrCheckBuffers(send, reply);
  }

  // send data and get reply
  void dispatch() {
    int slot = -1;
    std::unordered_map<std::string, torch::Tensor> sendBuffers =
        dc_->getSlot(&slot);
    assert(slot >= 0 && slot < dc_->batchsize);
    utils::copyTensors(sendTensors_, sendBuffers);

    dc_->markSlotFilled(slot);

    // std::cout << "dispatcher: trying to get reply" << std::endl;
    std::unordered_map<std::string, torch::Tensor> replyBuffers =
        dc_->getReply(slot);
    utils::copyTensors(replyBuffers, replyTensors_);

    dc_->releaseSlot(slot);
  }

 private:
  std::shared_ptr<DataChannel> dc_;
  std::unordered_map<std::string, torch::Tensor> sendTensors_;
  std::unordered_map<std::string, torch::Tensor> replyTensors_;
};
}
