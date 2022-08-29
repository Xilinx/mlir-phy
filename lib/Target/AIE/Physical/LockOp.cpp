//===- LockOp.cpp -----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "phy/Target/AIE/Physical/LockOp.h"

#include "phy/Dialect/Physical/PhysicalDialect.h"
#include "phy/Rewrite/RemoveOp.h"

#include "aie/AIEDialect.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace phy::physical;
using namespace phy::target::aie;
using namespace xilinx;

class LockOpToAieLowering : public OpConversionPattern<LockOp> {
  AIELoweringPatternSets *lowering;
  using OpAdaptor = typename LockOp::Adaptor;

public:
  LockOpToAieLowering(mlir::MLIRContext *context,
                      AIELoweringPatternSets *lowering)
      : OpConversionPattern<LockOp>(context), lowering(lowering) {}

  mlir::LogicalResult
  matchAndRewrite(LockOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tile = lowering->getTile(op);
    auto id = lowering->getId(op);

    auto lock = lowering->getLock(tile, id);
    rewriter.eraseOp(op);
    // AIE.useLock(%0, state, Release)
    rewriter.setInsertionPointAfter(lock);
    rewriter.create<xilinx::AIE::UseLockOp>(rewriter.getUnknownLoc(), lock,
                                            op.state(),
                                            xilinx::AIE::LockAction::Release);

    return success();
  }
};

void LockOpLoweringPatternSet::populatePatternSet(
    mlir::RewritePatternSet &patterns) {

  patterns.add<LockOpToAieLowering>(patterns.getContext(), lowering);

  // TODO: translate lock actions into AIE.useLock.
  patterns.add<OpRemover<LockAcquireOp>>(patterns.getContext());
  patterns.add<OpRemover<LockReleaseOp>>(patterns.getContext());
}
