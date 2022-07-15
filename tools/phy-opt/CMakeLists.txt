##===- CMakeLists.txt - phy-opt cmake  ------------------------*- cmake -*-===//
##
## Configure the phy-opt build.
##
## This file is licensed under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
##===----------------------------------------------------------------------===//

set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_tool(phy-opt
  phy-opt.cpp
)

llvm_update_compile_flags(phy-opt)

target_link_libraries(phy-opt
  PRIVATE
  Phy
  MLIRIR
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIROptLib
  MLIRParser
  MLIRFuncDialect
  MLIRSupport
  MLIRTransforms
  MLIRSCFDialect
)
