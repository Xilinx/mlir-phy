//===- CoreOp.cpp -----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "phy/Target/AIE/Physical/CoreOp.h"
#include "phy/Dialect/Physical/PhysicalDialect.h"

#include "aie/AIEDialect.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace phy::physical;
using namespace phy::target::aie;
using namespace xilinx;

class CoreOpToAieLowering : public OpConversionPattern<CoreOp> {
  AIELoweringPatternSets *lowering;
  using OpAdaptor = typename CoreOp::Adaptor;

public:
  CoreOpToAieLowering(mlir::MLIRContext *context,
                      AIELoweringPatternSets *lowering)
      : OpConversionPattern<CoreOp>(context), lowering(lowering) {}

  mlir::LogicalResult
  matchAndRewrite(CoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto Op = op.getOperation();
    rewriter.eraseOp(Op);
    return success();
  }
};

void CoreOpLoweringPatternSet::populatePatternSet(
    mlir::RewritePatternSet &patterns) {

  patterns.add<CoreOpToAieLowering>(patterns.getContext(), lowering);
}
