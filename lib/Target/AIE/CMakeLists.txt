##===- CMakeLists.txt - AIE target cmake  ---------------------*- cmake -*-===//
##
## This file is licensed under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
##===----------------------------------------------------------------------===//

if(AIE_FOUND)

add_phy_library(TargetAIE
  Affinity.cpp
  TargetResources.cpp

  LINK_LIBS
  Connectivity
)

endif()