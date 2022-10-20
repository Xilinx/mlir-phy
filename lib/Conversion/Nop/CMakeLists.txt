##===- CMakeLists.txt - nop pass cmake  -----------------------*- cmake -*-===//
##
## This file is licensed under the MIT License.
## SPDX-License-Identifier: MIT
##
## Copyright (C) 2022, Advanced Micro Devices, Inc.
##
##===----------------------------------------------------------------------===//

add_phy_conversion_library(Nop
  Nop.cpp

  DEPENDS
  PhyConnectivity
  MLIRPhyConversionPassIncGen

  LINK_LIBS
  MLIRPass
)