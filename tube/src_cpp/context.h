// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include <atomic>
#include <memory>
#include <thread>
#include <vector>

#include "env_thread.h"

namespace tube {

class Context {
 public:
  Context()
      : started_(false)
      , numTerminatedThread_(0) {
  }

  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;

  int pushEnvThread(std::shared_ptr<EnvThread> env) {
    assert(!started_);
    envs_.push_back(std::move(env));
    return (int)envs_.size();
  }

  void start() {
    for (int i = 0; i < (int)envs_.size(); ++i) {
      // std::thread t(&EnvThread::mainLoop, envs_[i]);
      std::thread t([this, i]() {
        envs_[i]->mainLoop();
        ++numTerminatedThread_;
      });
      t.detach();
      // threads_.push_back(std::move(t));
    }
  }

  bool terminated() {
    // std::cout << ">>> " << numTerminatedThread_ << std::endl;
    return numTerminatedThread_ == (int)envs_.size();
  }

 private:
  bool started_;
  std::atomic<int> numTerminatedThread_;
  std::vector<std::shared_ptr<EnvThread>> envs_;
  // std::vector<std::thread> threads_;
};
}
