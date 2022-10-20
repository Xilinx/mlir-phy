##===- CMakeLists.txt - connectivity cmake  -------------------*- cmake -*-===//
##
## This file is licensed under the MIT License.
## SPDX-License-Identifier: MIT
##
## Copyright (C) 2022, Advanced Micro Devices, Inc.
##
##===----------------------------------------------------------------------===//

file(GLOB_RECURSE Connectivity_Src
  "*.cpp"
)

add_phy_library(PhyConnectivity
  ${Connectivity_Src}
  LINK_LIBS
)
