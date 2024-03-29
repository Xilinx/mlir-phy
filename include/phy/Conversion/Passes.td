//===- Passes.td -------------------------------------------*- tablegen -*-===//
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
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
  let summary = "Convert physical dialect to AIE dialect";
  let description = [{
    This pass analyzes 'physical' dialect and lower it to the AIE dialect.
  }];
  let constructor = "phy::createPhysicalToAie()";
  let dependentDialects = [
    "phy::physical::PhysicalDialect",
    "xilinx::AIE::AIEDialect",
    "mlir::cf::ControlFlowDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect"
  ];
}

#endif // BUILD_WITH_AIE

//===----------------------------------------------------------------------===//
// LayoutToPhysical
//===----------------------------------------------------------------------===//

def LayoutToPhysical : Pass<"convert-layout-to-physical",
                            "mlir::ModuleOp"> {

  let summary = "Convert spatial with physical layout to physical dialect";
  let description = [{
    This pass analyzes 'layout.device' and the nested placed and routed
    elements on physical slots, lowers the elements in the specified device to
    the physical dialect.
  }];
  let constructor = "phy::createLayoutToPhysical()";
  let dependentDialects = [
    "phy::layout::LayoutDialect",
    "phy::physical::PhysicalDialect",
    "phy::spatial::SpatialDialect"
  ];
  let options = [
    Option<"device_option", "device", "std::string", /*default=*/"",
           "The device to be lowered">,
  ];
}

#endif // CIRCT_CONVERSION_PASSES_TD
