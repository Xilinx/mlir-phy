//===- ResourceTests.cpp - physical resource unit tests -------------------===//
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "phy/Connectivity/Resource.h"
#include "gtest/gtest.h"

using namespace phy::connectivity;

namespace {

TEST(Resource, Parses) {
  Resource r("tile/1.1/bank/0/buffer");
  EXPECT_EQ(r.key, "buffer");
  EXPECT_EQ(r.metadata["tile"], "1.1");
  EXPECT_EQ(r.metadata["bank"], "0");
}

TEST(Resource, Serializes) {
  Resource r("buffer", {{"tile", "1.1"}});
  EXPECT_EQ(r.toString(), "tile/1.1/buffer");
}

} // namespace
