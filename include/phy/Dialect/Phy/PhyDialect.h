//===- PhyDialect.h ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PHY_H
#define MLIR_DIALECT_PHY_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/CallInterfaces.h"

#include "phy/Dialect/Phy/PhyDialect.h.inc"
#include "phy/Dialect/Phy/PhyEnums.h.inc"
#include "phy/Dialect/Phy/PhyPasses.h.inc"

#define GET_TYPEDEF_CLASSES
#include "phy/Dialect/Phy/PhyTypes.h.inc"

#define GET_OP_CLASSES
#include "phy/Dialect/Phy/Phy.h.inc"

#endif // MLIR_DIALECT_PHY_H
