//===- Core.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "phy/Connectivity/Implementation/Core.h"

#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace phy;
using namespace phy::connectivity;

mlir::Operation *CoreImplementation::createOperation() {
  assert(node.getOperation() && "a core must be associated with a node");

  auto builder = OpBuilder::atBlockEnd(context.module.getBody());
  return builder.create<physical::CoreOp>(
      builder.getUnknownLoc(), physical::CoreType::get(builder.getContext()),
      node.callee(), node.operands());
}

void CoreImplementation::addSpatialOperation(mlir::Operation *spatial) {
  if (auto node_op = dyn_cast<spatial::NodeOp>(spatial)) {
    assert(!node.getOperation() && "a core can only hold a node");
    node = node_op;
  } else {
    assert("a core can only implement a node");
  }
}
