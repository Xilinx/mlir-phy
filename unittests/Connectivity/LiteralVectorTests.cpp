//===- LiteralVectorTests.cpp - physical resource unit tests --------------===//
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "phy/Connectivity/Serialization/LiteralVector.h"

#include <vector>

#include "gtest/gtest.h"

using namespace phy::connectivity;

namespace {

TEST(LiteralVector, Parses) {
  LiteralVector<int> r("1.1.2");
  EXPECT_EQ(r.vec()[0], 1);
  EXPECT_EQ(r.vec()[1], 1);
  EXPECT_EQ(r.vec()[2], 2);
}

TEST(LiteralVector, Serializes) {
  std::vector<int> v{1, 2, 3};
  LiteralVector<int> r(v);
  EXPECT_EQ(r.str(), "1.2.3");
}

} // namespace
