##===- CMakeLists.txt - AIE target cmake  ---------------------*- cmake -*-===//
##
## This file is licensed under the MIT License.
## SPDX-License-Identifier: MIT
##
## Copyright (C) 2022, Advanced Micro Devices, Inc.
##
##===----------------------------------------------------------------------===//

if(AIE_FOUND)

file(GLOB_RECURSE TargetAIE_Src
  "*.cpp"
)

add_phy_library(PhyTargetAIE
  ${TargetAIE_Src}
  LINK_LIBS
  PhyConnectivity
  PhyRewrite
)

endif()