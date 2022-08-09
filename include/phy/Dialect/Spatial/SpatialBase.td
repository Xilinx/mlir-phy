//===- SpatialBase.td --------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"

#ifndef MLIR_PHY_DIALECT_SPATIAL_BASE_TD
#define MLIR_PHY_DIALECT_SPATIAL_BASE_TD

def Spatial_Dialect : Dialect {
  let name = "spatial";
  let cppNamespace = "::phy::spatial";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect"
  ];
  let description = [{
    This dialect defines a graph of stateful vertices of free-running compute
    nodes and message queues.  The vertices are connected using flow control 
    edges called flows.  This dialect is intended to be target-independent
    and express the logical concurrency semantics.  It does not express
    anything that is target-dependent.
  }];
  let useDefaultTypePrinterParser = 1;
}

#endif  // MLIR_PHY_DIALECT_SPATIAL_BASE_TD
