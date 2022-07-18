//===- PhyDialect.cpp - Implement the Phy dialect -------------------===//
//
// This file implements the Phy dialect.
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "phy/Dialect/Phy/PhyDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mlir;
using namespace ::phy;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

void PhyDialect::initialize() {
  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "phy/Dialect/Phy/Phy.cpp.inc"
      >();
  // Register types.
  addTypes<
#define GET_TYPEDEF_LIST
#include "phy/Dialect/Phy/PhyTypes.cpp.inc"
      >();
}

// Provide implementations for the enums, attributes and interfaces that we use.
#include "phy/Dialect/Phy/PhyDialect.cpp.inc"
#include "phy/Dialect/Phy/PhyEnums.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "phy/Dialect/Phy/PhyTypes.cpp.inc"

// TableGen'd op method definitions
#define GET_OP_CLASSES
#include "phy/Dialect/Phy/Phy.cpp.inc"