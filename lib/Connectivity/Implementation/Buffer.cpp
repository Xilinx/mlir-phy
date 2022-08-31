//===- Buffer.cpp -----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "phy/Connectivity/Implementation/Buffer.h"

#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace phy;
using namespace phy::connectivity;

mlir::Operation *BufferImplementation::createOperation() {
  assert(queue.getOperation() && "a buffer must be associated with a queue");

  auto builder = OpBuilder::atBlockEnd(context.module.getBody());
  return builder.create<physical::BufferOp>(
      builder.getUnknownLoc(),
      queue.queue().getType().dyn_cast<spatial::QueueType>().getDatatype());
}

void BufferImplementation::addSpatialOperation(mlir::Operation *spatial) {
  if (auto queue_op = dyn_cast<spatial::QueueOp>(spatial)) {
    assert(!queue.getOperation() && "a buffer can only hold a queue");
    queue = queue_op;
  } else {
    assert("a buffer can only implement a queue");
  }
}
