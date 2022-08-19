//===- ImplementationTests.cpp - physical implementation unit tests -------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "phy/Connectivity/Implementation.h"
#include "gtest/gtest.h"

using namespace phy::connectivity;

namespace {

TEST(Implementation, Parses) {
  Implementation i("tile/1.1/bank/0/buffer,tile/1.1/lock");
  EXPECT_EQ(i.phys.front().key, "buffer");
  EXPECT_EQ(i.phys.front().metadata["tile"], "1.1");
  EXPECT_EQ(i.phys.front().metadata["bank"], "0");
  EXPECT_EQ(i.phys.back().key, "lock");
  EXPECT_EQ(i.phys.back().metadata["tile"], "1.1");
}

TEST(Implementation, Serializes) {
  Implementation i;
  i.phys.push_back(PhysicalResource("buffer", {{"tile", "1.1"}}));
  i.phys.push_back(PhysicalResource("lock", {{"tile", "1.1"}}));
  EXPECT_EQ(i.toString(), "tile/1.1/buffer,tile/1.1/lock");
}

} // namespace
