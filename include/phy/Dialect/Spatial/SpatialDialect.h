//===- SpatialDialect.h -----------------------------------------*- C++ -*-===//
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_DIALECT_SPATIAL_H
#define MLIR_PHY_DIALECT_SPATIAL_H

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

#include "phy/Dialect/Spatial/SpatialDialect.h.inc"
#include "phy/Dialect/Spatial/SpatialEnums.h.inc"
#include "phy/Dialect/Spatial/SpatialPasses.h.inc"

#define GET_TYPEDEF_CLASSES
#include "phy/Dialect/Spatial/SpatialTypes.h.inc"

#define GET_OP_CLASSES
#include "phy/Dialect/Spatial/Spatial.h.inc"

#endif // MLIR_PHY_DIALECT_SPATIAL_H
