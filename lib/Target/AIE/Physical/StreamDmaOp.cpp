//===- StreamDmaOp.cpp ------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "phy/Target/AIE/Physical/StreamDmaOp.h"

#include "phy/Dialect/Physical/PhysicalDialect.h"
#include "phy/Target/AIE/TargetResources.h"

#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace phy::physical;
using namespace phy::target::aie;
using namespace xilinx;

class StreamDmaOpToAieLowering : public OpConversionPattern<StreamDmaOp> {
  AIELoweringPatternSets *lowering;
  using OpAdaptor = typename StreamDmaOp::Adaptor;

public:
  StreamDmaOpToAieLowering(mlir::MLIRContext *context,
                           AIELoweringPatternSets *lowering)
      : OpConversionPattern<StreamDmaOp>(context), lowering(lowering) {}

  mlir::LogicalResult
  matchAndRewrite(StreamDmaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto tile = lowering->getTileIndex(op);

    if (TargetResources().isShimTile(tile.first, tile.second)) {
      lowering->getShimDma(tile);
    } else {
      lowering->getDma(tile);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

void StreamDmaOpLoweringPatternSet::populatePatternSet(
    mlir::RewritePatternSet &patterns) {

  patterns.add<StreamDmaOpToAieLowering>(patterns.getContext(), lowering);
}
