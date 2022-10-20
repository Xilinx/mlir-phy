##===- CMakeLists.txt - physical dialect cmake  ---------------*- cmake -*-===//
##
## This file is licensed under the MIT License.
## SPDX-License-Identifier: MIT
##
## Copyright (C) 2022, Advanced Micro Devices, Inc.
##
##===----------------------------------------------------------------------===//

add_phy_dialect(Physical physical)

set(LLVM_TARGET_DEFINITIONS Physical.td)
phy_tablegen(PhysicalEnums.h.inc -gen-enum-decls)
phy_tablegen(PhysicalEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRPhysicalEnumsIncGen)
add_phy_doc(Physical PhysicalDialect -gen-dialect-doc -dialect physical)
add_dependencies(phy-headers MLIRPhysicalEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS PhysicalPasses.td)
phy_tablegen(PhysicalPasses.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRPhysicalTransformsIncGen)
add_phy_doc(PhysicalPasses PhysicalPasses -gen-pass-doc)
add_dependencies(phy-headers MLIRPhysicalTransformsIncGen)
