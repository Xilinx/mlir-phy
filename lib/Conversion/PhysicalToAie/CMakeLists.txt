##===- CMakeLists.txt - convert-phy-to-aie cmake  -------------*- cmake -*-===//
##
## This file is licensed under the MIT License.
## SPDX-License-Identifier: MIT
##
## Copyright (C) 2022, Advanced Micro Devices, Inc.
##
##===----------------------------------------------------------------------===//

if(AIE_FOUND)

add_phy_conversion_library(PhysicalToAie
  PhysicalToAie.cpp

  DEPENDS
  MLIRPhyConversionPassIncGen

  LINK_LIBS
  MLIRPass
  AIE
  PhyTargetAIE
)

endif()
