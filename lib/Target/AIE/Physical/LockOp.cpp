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
#include "phy/Support/LexicalCast.h"

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

    // %0 = AIE.lock(tile, id)
    auto lock = lowering->locks[op] =
        rewriter.replaceOpWithNewOp<xilinx::AIE::LockOp>(op, tile, id);

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

  // Remove functions with lock type as it inputs.
  // If the function is not fully inlined, the pass will fail.
  patterns.add<OpRemover<func::FuncOp>>(patterns.getContext());

  // TODO: translate lock actions into AIE.useLock.
  patterns.add<OpRemover<LockAcquireOp>>(patterns.getContext());
  patterns.add<OpRemover<LockReleaseOp>>(patterns.getContext());
}

void LockOpLoweringPatternSet::populateTarget(mlir::ConversionTarget &target) {
  target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
    // function calls cannot have lock arguments
    for (auto input : op.getCalleeType().getInputs())
      if (input.isa<LockType>())
        return false;
    return true;
  });

  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    // functions cannot have lock type inputs
    for (auto input : op.getFunctionType().getInputs())
      if (input.isa<LockType>())
        return false;
    return true;
  });

  target.addIllegalOp<LockOp, LockAcquireOp, LockReleaseOp>();
}
