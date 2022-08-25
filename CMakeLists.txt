##===- CMakeLists.txt - PHY cmake root ------------------------*- cmake -*-===//
##
## Configure the PHY build.
## Adapted from llvm/PHY/CMakeLists.txt
##
## This file is licensed under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
##===----------------------------------------------------------------------===//

cmake_minimum_required(VERSION 3.13.4)

if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

# CMP0116: Ninja generators transform `DEPFILE`s from `add_custom_command()`
# New in CMake 3.20. https://cmake.org/cmake/help/latest/policy/CMP0116.html
if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# If we are not building as a part of LLVM, build PHY as an
# standalone project, using LLVM as an external library:
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)

#-------------------------------------------------------------------------------
# Project setup and globals
#-------------------------------------------------------------------------------
  project(PHY LANGUAGES CXX C)

#-------------------------------------------------------------------------------
# Options and settings
#-------------------------------------------------------------------------------
  option(LLVM_INCLUDE_TOOLS "Generate build targets for the LLVM tools." ON)
  option(LLVM_BUILD_TOOLS "Build the LLVM tools. If OFF, just generate build targets." ON)

  if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHs-c- /GR-")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions -fno-rtti")
  endif()

  # Enable address sanitizer in debug mode
  if (CMAKE_BUILD_TYPE EQUAL "DEBUG")
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
  endif()

#-------------------------------------------------------------------------------
# MLIR/LLVM Configuration
#-------------------------------------------------------------------------------

  find_package(MLIR REQUIRED CONFIG)

  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)

  set(PHY_BUILT_STANDALONE 1)
  set(BACKEND_PACKAGE_STRING "LLVM ${LLVM_PACKAGE_VERSION}")

else()

  set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir ) # --src-root
  set(MLIR_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include ) # --includedir
  set(MLIR_TABLEGEN_OUTPUT_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
  set(MLIR_TABLEGEN_EXE $<TARGET_FILE:mlir-tblgen>)
  include_directories(SYSTEM ${MLIR_INCLUDE_DIR})
  include_directories(SYSTEM ${MLIR_TABLEGEN_OUTPUT_DIR})

  set(BACKEND_PACKAGE_STRING "${PACKAGE_STRING}")

endif()

# Define the default arguments to use with 'lit', and an option for the user to
# override.
set(LIT_ARGS_DEFAULT "-sv")
if(MSVC_IDE OR XCODE)
  set(LIT_ARGS_DEFAULT "${LIT_ARGS_DEFAULT} --no-progress-bar")
endif()
set(LLVM_LIT_ARGS "${LIT_ARGS_DEFAULT}" CACHE STRING "Default options for lit")

#-------------------------------------------------------------------------------
# PHY configuration
#-------------------------------------------------------------------------------

# PHY project.
set(PHY_MAIN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR} ) # --src-root
set(PHY_MAIN_INCLUDE_DIR ${PHY_MAIN_SRC_DIR}/include)

set(PHY_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PHY_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(PHY_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include)
set(PHY_LIBRARY_DIR ${CMAKE_BINARY_DIR}/lib)
set(PHY_TOOLS_DIR ${CMAKE_BINARY_DIR}/bin)
set(PHY_UTILS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/utils)
set(PHY_PYTHON_PACKAGES_DIR ${PHY_BINARY_DIR}/python_packages)

list(APPEND CMAKE_MODULE_PATH "${MLIR_MAIN_SRC_DIR}/cmake/modules")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
include(AddPhy)

# Installing the headers and docs needs to depend on generating any public
# tablegen'd targets.
add_custom_target(phy-headers)
set_target_properties(phy-headers PROPERTIES FOLDER "Misc")
add_custom_target(phy-doc)

# Add MLIR and LLVM headers to the include path
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

# Add PHY files to the include path
include_directories(${PHY_MAIN_INCLUDE_DIR})
include_directories(${PHY_INCLUDE_DIR})

#-------------------------------------------------------------------------------
# Dependency configuration
#-------------------------------------------------------------------------------

find_package(AIE)
if(AIE_FOUND)
  message(STATUS "Building with MLIR-AIE")
  message(STATUS "Using AIEConfig.cmake in: ${AIE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${AIE_CMAKE_DIR}")

  add_phy_definition(-DBUILD_WITH_AIE)
  link_directories(${AIE_BINARY_DIR}/lib)
  include_directories(${AIE_INCLUDE_DIRS})
endif()

#-------------------------------------------------------------------------------
# Directory setup
#-------------------------------------------------------------------------------

add_subdirectory(include/phy)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(unittests)
add_subdirectory(test)

option(PHY_INCLUDE_DOCS "Generate build targets for the Phy docs.")
if(PHY_INCLUDE_DOCS)
  add_subdirectory(docs)
endif()

install(DIRECTORY include/phy
  DESTINATION include
  COMPONENT phy-headers
  FILES_MATCHING
  PATTERN "*.def"
  PATTERN "*.h"
  PATTERN "*.inc"
  PATTERN "*.td"
  PATTERN "LICENSE.TXT"
)

install(DIRECTORY ${PHY_INCLUDE_DIR}/phy
  DESTINATION include
  COMPONENT phy-headers
  FILES_MATCHING
  PATTERN "*.def"
  PATTERN "*.h"
  PATTERN "*.gen"
  PATTERN "*.inc"
  PATTERN "*.td"
  PATTERN "CMakeFiles" EXCLUDE
  PATTERN "config.h" EXCLUDE
)

if(NOT LLVM_ENABLE_IDE)
  add_llvm_install_targets(install-phy-headers
                           DEPENDS phy-headers
                           COMPONENT phy-headers)
endif()

add_subdirectory(cmake/modules)
