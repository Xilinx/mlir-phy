//===- PhysicalDialect.h ----------------------------------------*- C++ -*-===//
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_DIALECT_PHYSICAL_H
#define MLIR_PHY_DIALECT_PHYSICAL_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

#include "phy/Dialect/Physical/PhysicalDialect.h.inc"
#include "phy/Dialect/Physical/PhysicalEnums.h.inc"
#include "phy/Dialect/Physical/PhysicalPasses.h.inc"

#define GET_TYPEDEF_CLASSES
#include "phy/Dialect/Physical/PhysicalTypes.h.inc"

#define GET_OP_CLASSES
#include "phy/Dialect/Physical/Physical.h.inc"

#endif // MLIR_PHY_DIALECT_PHYSICAL_H
