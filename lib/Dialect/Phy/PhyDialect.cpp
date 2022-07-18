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

// Ops verifiers
LogicalResult PeOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!fnAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");

  func::FuncOp fn =
      symbolTable.lookupNearestSymbolFrom<func::FuncOp>(*this, fnAttr);
  if (!fn) {
    return emitOpError() << "expected symbol reference " << callee()
                         << " to point to a function";
  }

  // Verify that the operand and result types match the callee.
  auto fnType = fn.getFunctionType();
  if (fnType.getNumInputs() != getNumOperands())
    return emitOpError("incorrect number of operands for callee");

  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
    if (getOperand(i).getType() != fnType.getInput(i))
      return emitOpError("operand type mismatch: expected operand type ")
             << fnType.getInput(i) << ", but provided "
             << getOperand(i).getType() << " for operand number " << i;

  if (fnType.getNumResults() != 0)
    return emitOpError("callee cannot have a return value");

  return success();
}

LogicalResult BusOp::verify() {
  auto busType = (*this).bus().getType().cast<BusType>().getDatatype();

  // Verify that all endpoints have the same base type as the bus
  for (auto endpoint : (*this).endpoints()) {
    if (auto memref = endpoint.getType().dyn_cast<MemRefType>()) {
      if (memref.getElementType() != busType) {
        return emitOpError("endpoints must have the same base type as the bus")
               << ": expected endpoint element type " << busType
               << ", but provided " << memref.getElementType();
      }
    }
  }

  return success();
}