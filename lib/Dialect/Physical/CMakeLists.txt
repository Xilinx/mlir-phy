##===- CMakeLists.txt - physical dialect cmake  ---------------*- cmake -*-===//
##
## This file is licensed under the MIT License.
## SPDX-License-Identifier: MIT
##
## Copyright (C) 2022, Advanced Micro Devices, Inc.
##
##===----------------------------------------------------------------------===//

set(PHYSICAL_SRCS
  PhysicalDialect.cpp
  PhysicalOps.cpp
)

add_phy_dialect_library(Physical
  ${PHYSICAL_SRCS}

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR

  DEPENDS
  MLIRPhysicalEnumsIncGen
)
