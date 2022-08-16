##===- CMakeLists.txt - unit tests cmake  ---------------------*- cmake -*-===//
##
## This file is licensed under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
##===----------------------------------------------------------------------===//

add_custom_target(PhyUnitTests)
set_target_properties(PhyUnitTests PROPERTIES FOLDER "PHY Tests")

function(add_phy_unittest test_dirname)
  add_unittest(PhyUnitTests ${test_dirname} ${ARGN})
endfunction()

add_subdirectory(Connectivity)
