//===- SpatialOps.cpp - Implement the Phy operations ----------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "phy/Dialect/Spatial/SpatialDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mlir;
using namespace ::phy::spatial;

LogicalResult NodeOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
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

LogicalResult FlowOp::verify() {
  Type srcType = src().getType();
  Type destType = dest().getType();

  if (srcType.isa<NodeType>() && destType.isa<NodeType>())
    return emitOpError("a node cannot be connected to a node using a flow");

  if (srcType.isa<QueueType>() && destType.isa<QueueType>())
    return emitOpError("a queue cannot be connected to a queue using a flow");

  Type datatype;
  if (auto srcQueue = srcType.dyn_cast<QueueType>())
    datatype = srcQueue.getDatatype();
  else if (auto destQueue = destType.dyn_cast<QueueType>())
    datatype = destQueue.getDatatype();
  else
    return emitOpError("one endpoint of the flow must be a queue");

  if (flow().getType().dyn_cast<FlowType>().getDatatype() != datatype)
    return emitOpError("the datatype of the flow must match the queue");

  return success();
}
