//===- StreamOp.cpp ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "phy/Target/AIE/Physical/StreamOp.h"

#include "phy/Dialect/Physical/PhysicalDialect.h"

#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace phy::physical;
using namespace phy::target::aie;
using namespace xilinx;

class StreamOpToAieLowering : public OpConversionPattern<StreamOp> {
  AIELoweringPatternSets *lowering;
  using OpAdaptor = typename StreamOp::Adaptor;

public:
  StreamOpToAieLowering(mlir::MLIRContext *context,
                        AIELoweringPatternSets *lowering)
      : OpConversionPattern<StreamOp>(context), lowering(lowering) {}

  mlir::LogicalResult
  matchAndRewrite(StreamOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

void StreamOpLoweringPatternSet::populatePatternSet(
    mlir::RewritePatternSet &patterns) {

  patterns.add<StreamOpToAieLowering>(patterns.getContext(), lowering);
}
