//===- LayoutOps.cpp - Implement the Layout operations --------------------===//
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "phy/Dialect/Layout/LayoutDialect.h"
#include "phy/Dialect/Spatial/SpatialDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace ::mlir;
using namespace ::phy::layout;
using namespace ::phy::spatial;

LogicalResult RouteOp::verify() {
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

  return success();
}
