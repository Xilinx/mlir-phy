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
// Nop
//===----------------------------------------------------------------------===//

def Nop : Pass<"convert-nop", "mlir::ModuleOp"> {
  let summary = "No operation";
  let description = [{
    This pass does nothing.  It simply relies on all dialects and is used to
    keep building in case no other conversion has its dependency satisfied.
  }];
  let constructor = "phy::createNop()";
  let dependentDialects = [
    "phy::layout::LayoutDialect",
    "phy::physical::PhysicalDialect",
    "phy::spatial::SpatialDialect",
  ];
}

//===----------------------------------------------------------------------===//
// PhysicalToAie
//===----------------------------------------------------------------------===//

#ifdef BUILD_WITH_AIE

def PhysicalToAie : Pass<"convert-physical-to-aie", "mlir::ModuleOp"> {
  let summary = "Convert Physical dialect to AIE dialect";
  let description = [{
    This pass analyzes 'physical.device' and the nested placed and routed
    elements, lowers the elements in the 'aie' device to the AIE dialect.
  }];
  let constructor = "phy::createPhysicalToAie()";
  let dependentDialects = [
    "phy::physical::PhysicalDialect",
    "xilinx::AIE::AIEDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect"
  ];
}

#endif // BUILD_WITH_AIE

#endif // CIRCT_CONVERSION_PASSES_TD