// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#pragma once

namespace detail {

struct NonCopyable {
  NonCopyable(const NonCopyable&) = delete;
  NonCopyable& operator=(const NonCopyable&) = delete;
};

struct NonMovable {
  NonMovable(NonMovable&&) = delete;
  NonMovable&& operator=(NonMovable&&) = delete;
};

} // namespace detail
