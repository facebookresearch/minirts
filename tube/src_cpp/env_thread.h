// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

namespace tube {

class EnvThread {
 public:
  virtual ~EnvThread() {
  }

  virtual void mainLoop() = 0;
};
}
