##===- CMakeLists.txt - convert-phy-to-aie cmake  -------------*- cmake -*-===//
##
## This file is licensed under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
