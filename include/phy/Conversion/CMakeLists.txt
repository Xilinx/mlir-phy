##===- CMakeLists.txt - phy passes cmake  --------------------*- cmake -*-===//
##
## This file is licensed under the MIT License.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: MIT
##
## Copyright (C) 2022, Advanced Micro Devices, Inc.
##
##===----------------------------------------------------------------------===//

set(LLVM_TARGET_DEFINITIONS Passes.td)
phy_tablegen(Passes.h.inc -gen-pass-decls -name Conversion)
add_public_tablegen_target(MLIRPhyConversionPassIncGen)
add_phy_doc(Passes PhyConversionPass -gen-pass-doc)
