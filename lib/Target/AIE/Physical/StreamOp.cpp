//===- StreamOp.cpp ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
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
  using OpAdaptor = typename StreamOp::Adaptor;

public:
  StreamOpToAieLowering(mlir::MLIRContext *context,
                        AIELoweringPatternSets *lowering)
      : OpConversionPattern<StreamOp>(context) {}

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
