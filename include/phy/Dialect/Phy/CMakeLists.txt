##===- CMakeLists.txt - phy dialect cmake  --------------------*- cmake -*-===//
##
## This file is licensed under the Apache License v2.0 with LLVM Exceptions.
## See https://llvm.org/LICENSE.txt for license information.
## SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
##
##===----------------------------------------------------------------------===//

add_phy_dialect(Phy phy)
add_phy_dialect_doc(Phy phy)

set(LLVM_TARGET_DEFINITIONS Phy.td)
mlir_tablegen(PhyEnums.h.inc -gen-enum-decls)
mlir_tablegen(PhyEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRPhyEnumsIncGen)
add_phy_doc(Phy PhyDialect -gen-dialect-doc)
add_dependencies(phy-headers MLIRPhyEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS PhyPasses.td)
mlir_tablegen(PhyPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRPhyTransformsIncGen)
add_phy_doc(PhyPasses PhyPasses -gen-pass-doc)
add_dependencies(phy-headers MLIRPhyTransformsIncGen)
