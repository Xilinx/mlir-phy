##===- CMakeLists.txt - phy dialect cmake  --------------------*- cmake -*-===//
##
## This file is licensed under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
##===----------------------------------------------------------------------===//

set(PHY_SRCS
  PhyDialect.cpp
  PhyOps.cpp
)

add_phy_dialect_library(Phy
  ${PHY_SRCS}

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR

  DEPENDS
  MLIRPhyEnumsIncGen
)
