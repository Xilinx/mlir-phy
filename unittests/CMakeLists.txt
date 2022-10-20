##===- CMakeLists.txt - unit tests cmake  ---------------------*- cmake -*-===//
##
## This file is licensed under the MIT License.
## SPDX-License-Identifier: MIT
##
## Copyright (C) 2022, Advanced Micro Devices, Inc.
##
##===----------------------------------------------------------------------===//

add_custom_target(PhyUnitTests)
set_target_properties(PhyUnitTests PROPERTIES FOLDER "PHY Tests")

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_subdirectory(Connectivity)
