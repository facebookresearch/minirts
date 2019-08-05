// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
#include "ai_factory.h"

std::map<std::string, AIFactory::RegFunc> AIFactory::_factories;
