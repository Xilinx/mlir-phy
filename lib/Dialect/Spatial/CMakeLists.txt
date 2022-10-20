##===- CMakeLists.txt - spatial dialect cmake  ----------------*- cmake -*-===//
##
## This file is licensed under the MIT License.
## SPDX-License-Identifier: MIT
##
## Copyright (C) 2022, Advanced Micro Devices, Inc.
##
##===----------------------------------------------------------------------===//

set(SPATIAL_SRCS
  SpatialDialect.cpp
  SpatialOps.cpp
)

add_phy_dialect_library(Spatial
  ${SPATIAL_SRCS}

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR

  DEPENDS
  MLIRSpatialEnumsIncGen
)
