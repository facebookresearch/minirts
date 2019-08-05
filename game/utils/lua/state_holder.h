// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

#include "common.h"
#include "selene.h"

class StateHolder : public detail::NonMovable, detail::NonCopyable {
 public:
  static sel::State& GetState() {
    // This initialization should be thread safe according to the standard.
    static thread_local sel::State state{true};
    return state;
  }

 protected:
  StateHolder() = default;
  ~StateHolder() = default;
};
