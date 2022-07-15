//===- Phy.td ----------------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"

#ifndef MLIR_DIALECT_PHY_TD
#define MLIR_DIALECT_PHY_TD

def Phy_Dialect : Dialect {
  let name = "phy";
  let cppNamespace = "::phy";
  let dependentDialects = ["::mlir::func::FuncDialect"];
  let description = [{
    This dialect defines a flattened graph of elements including free-running
    processing elements (PEs), the memory that stores or registers data
    (buffers), message-passing like streams or packets (networks), and memory
    access with addresses (buses) that transmits data, with spatial
    information, which is the connections and numbers of elements.
    The elements defined in this dialect can be mapped to a physical location
    or a set of wires of the devices.
  }];
}

include "phy/Dialect/Phy/PhyTypes.td"
include "phy/Dialect/Phy/PhyOps.td"

#endif  // MLIR_DIALECT_PHY_TD
