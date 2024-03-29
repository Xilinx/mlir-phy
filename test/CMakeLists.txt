##===- CMakeLists.txt - LIT Testing ---------------------------*- cmake -*-===//
##
## Configure the MLIR-PHY tests.
##
## This file is licensed under the MIT License.
## SPDX-License-Identifier: MIT
##
## Copyright (C) 2022, Advanced Micro Devices, Inc.
##
##===----------------------------------------------------------------------===//

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.cfg.py
)

set(PHY_TEST_DEPENDS
  FileCheck count not
  phy-opt
  phy-translate
)

list(APPEND PHY_TEST_DEPENDS PhyUnitTests)

add_lit_testsuite(check-phy
  "Running the MLIR-PHY regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${PHY_TEST_DEPENDS}
)
set_target_properties(check-phy PROPERTIES FOLDER "Tests")

add_lit_testsuites(PHY
  ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${PHY_TEST_DEPS}
)
