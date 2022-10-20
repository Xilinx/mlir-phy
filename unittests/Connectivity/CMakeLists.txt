##===- CMakeLists.txt - connectivity tests cmake  -------------*- cmake -*-===//
##
## This file is licensed under the MIT License.
## SPDX-License-Identifier: MIT
##
## Copyright (C) 2022, Advanced Micro Devices, Inc.
##
##===----------------------------------------------------------------------===//

add_executable(PHYConnectivityTests
  LiteralVectorTests.cpp
  ResourceListTests.cpp
  ResourceTests.cpp
)

target_link_libraries(PHYConnectivityTests
  PRIVATE
  PhyConnectivity
  GTest::gtest_main
)
