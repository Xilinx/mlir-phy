##===- CMakeLists.txt - layout dialect cmake  -----------------*- cmake -*-===//
##
## This file is licensed under the MIT License.
## SPDX-License-Identifier: MIT
##
## Copyright (C) 2022, Advanced Micro Devices, Inc.
##
##===----------------------------------------------------------------------===//

set(LAYOUT_SRCS
  LayoutDialect.cpp
  LayoutOps.cpp
)

add_phy_dialect_library(Layout
  ${LAYOUT_SRCS}

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/include

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR

  DEPENDS
  MLIRLayoutEnumsIncGen
)
