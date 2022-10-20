##===- CMakeLists.txt - convert-layout-to-physical cmake  -----*- cmake -*-===//
##
## This file is licensed under the MIT License.
## SPDX-License-Identifier: MIT
##
## Copyright (C) 2022, Advanced Micro Devices, Inc.
##
##===----------------------------------------------------------------------===//

add_phy_conversion_library(LayoutToPhysical
  LayoutToPhysical.cpp

  DEPENDS
  MLIRPhyConversionPassIncGen

  LINK_LIBS
  MLIRPass
  PhyConnectivity
)
