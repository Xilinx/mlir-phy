##===- CMakeLists.txt - rewrite cmake  ------------------------*- cmake -*-===//
##
## This file is licensed under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
##===----------------------------------------------------------------------===//

file(GLOB_RECURSE Rewrite_Src
  "*.cpp"
)

add_phy_library(PhyRewrite
  ${Rewrite_Src}
)
