##===- CMakeLists.txt - rewrite cmake  ------------------------*- cmake -*-===//
##
## This file is licensed under the MIT License.
## SPDX-License-Identifier: MIT
##
## Copyright (C) 2022, Advanced Micro Devices, Inc.
##
##===----------------------------------------------------------------------===//

file(GLOB_RECURSE Rewrite_Src
  "*.cpp"
)

add_phy_library(PhyRewrite
  ${Rewrite_Src}
)
