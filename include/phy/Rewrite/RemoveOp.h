//===- RemoveOp.h ------------------------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "phy/Dialect/Physical/PhysicalDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"

#ifndef MLIR_PHY_REWRITE_REMOVE_OP_H
#define MLIR_PHY_REWRITE_REMOVE_OP_H

namespace phy {
namespace rewrite {

template <typename Op>
class OpRemover : public mlir::OpConversionPattern<Op> {
  using OpAdaptor = typename Op::Adaptor;

public:
  OpRemover(mlir::MLIRContext *context) : OpConversionPattern<Op>(context) {}

  mlir::LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace rewrite
} // namespace phy

#endif // MLIR_PHY_REWRITE_REMOVE_OP_H
