//===- PhyOps.cpp - Implement the Phy operations --------------------------===//
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
// Ops verifiers
//===----------------------------------------------------------------------===//

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
  auto dataType = (*this).bus().getType().cast<BusType>().getDatatype();

  // Verify that all endpoints have the same base type as the bus
  for (auto endpoint : (*this).endpoints()) {
    if (auto memref = endpoint.getType().dyn_cast<MemRefType>()) {
      if (memref.getElementType() != dataType) {
        return emitOpError("endpoints must have the same base type as the bus")
               << ": expected endpoint element type " << dataType
               << ", but provided " << memref.getElementType();
      }
    }
  }

  return success();
}
