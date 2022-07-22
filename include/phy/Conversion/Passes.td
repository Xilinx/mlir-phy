//===- Passes.td -------------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef PHY_CONVERSION_PASSES_TD
#define PHY_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// PhyToAie
//===----------------------------------------------------------------------===//

#ifdef BUILD_WITH_AIE

def PhyToAie : Pass<"convert-phy-to-aie", "mlir::ModuleOp"> {
  let summary = "Convert Phy dialect to AIE dialect";
  let description = [{
    This pass analyzes 'phy.device' and the nested placed and routed elements,
    lowers the elements in the 'aie' device to the AIE dialect.
  }];
  let constructor = "phy::createPhyToAie()";
  let dependentDialects = [
    "phy::PhyDialect",
    "xilinx::AIE::AIEDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect"
  ];
}

#endif // BUILD_WITH_AIE

#endif // CIRCT_CONVERSION_PASSES_TD
