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

using namespace phy;

//===----------------------------------------------------------------------===//
// Dialect specification.
//===----------------------------------------------------------------------===//

void PhyDialect::initialize() {
  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "phy/Dialect/Phy/Phy.cpp.inc"
      >();
}

// Provide implementations for the enums, attributes and interfaces that we use.
#define GET_ATTRDEF_CLASSES
#include "phy/Dialect/Phy/PhyDialect.cpp.inc"
#include "phy/Dialect/Phy/PhyEnums.cpp.inc"
