//===- EndOps.td -------------------------------------------*- tablegen -*-===//
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_DIALECT_PHYSICAL_END_OPS_TD
#define MLIR_PHY_DIALECT_PHYSICAL_END_OPS_TD

include "phy/Dialect/Physical/PhysicalBase.td"

//===----------------------------------------------------------------------===//
// EndOp
//===----------------------------------------------------------------------===//

def EndOp : Physical_Op<"end", [Terminator]> {
  let summary = "end of usage of a region";
  let description = [{
    'physical.end' is an implied terminator.
  }];
  let assemblyFormat = "attr-dict";
}

#endif  // MLIR_PHY_DIALECT_PHYSICAL_END_OPS_TD
