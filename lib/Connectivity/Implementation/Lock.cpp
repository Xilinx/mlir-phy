//===- Lock.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
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

void LockImplementation::translateUserOperation(mlir::Value value,
                                                mlir::Operation *user) {

  OpBuilder builder(user);

  if (auto emplace = dyn_cast<spatial::EmplaceOp>(user)) {
    builder.create<physical::LockAcquireOp>(
        builder.getUnknownLoc(), builder.getI64IntegerAttr(0), value);

  } else if (auto front = dyn_cast<spatial::FrontOp>(user)) {
    builder.create<physical::LockAcquireOp>(
        builder.getUnknownLoc(), builder.getI64IntegerAttr(1), value);

  } else if (auto push = dyn_cast<spatial::PushOp>(user)) {
    builder.create<physical::LockReleaseOp>(
        builder.getUnknownLoc(), builder.getI64IntegerAttr(1), value);

  } else if (auto pop = dyn_cast<spatial::PopOp>(user)) {
    builder.create<physical::LockReleaseOp>(
        builder.getUnknownLoc(), builder.getI64IntegerAttr(0), value);
  }
}