//===- Lock.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "phy/Connectivity/Implementation/Lock.h"

#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace phy;
using namespace phy::connectivity;

mlir::Operation *LockImplementation::createOperation() {
  auto builder = OpBuilder::atBlockEnd(context.module.getBody());
  return builder.create<physical::LockOp>(
      builder.getUnknownLoc(), physical::LockType::get(builder.getContext()),
      builder.getI64IntegerAttr(0));
}
