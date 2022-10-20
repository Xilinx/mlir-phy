//===- SpatialDialect.cpp - Implement the Spatial dialect -----------------===//
//
// This file implements the Spatial dialect.
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "phy/Dialect/Spatial/SpatialDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mlir;
using namespace ::phy::spatial;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

void SpatialDialect::initialize() {
  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "phy/Dialect/Spatial/Spatial.cpp.inc"
      >();
  // Register types.
  addTypes<
#define GET_TYPEDEF_LIST
#include "phy/Dialect/Spatial/SpatialTypes.cpp.inc"
      >();
}

// Provide implementations for the enums, attributes and interfaces that we use.
#include "phy/Dialect/Spatial/SpatialDialect.cpp.inc"
#include "phy/Dialect/Spatial/SpatialEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "phy/Dialect/Spatial/SpatialTypes.cpp.inc"

// TableGen'd op method definitions
#define GET_OP_CLASSES
#include "phy/Dialect/Spatial/Spatial.cpp.inc"
