//===- Phy.td ----------------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

#ifndef PHY_TD_
#define PHY_TD_

def Phy_Dialect : Dialect {
  let name = "phy";
  let cppNamespace = "::phy";
  let description = [{
  }];
  let useDefaultTypePrinterParser = 1;
}

#endif  // PHY_TD_
