//===- BufferOp.cpp ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "phy/Target/AIE/Physical/BufferOp.h"

#include "phy/Dialect/Physical/PhysicalDialect.h"
#include "phy/Support/LexicalCast.h"

#include "aie/AIEDialect.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace phy::physical;
using namespace phy::target::aie;
using namespace xilinx;

class BufferOpToAieLowering : public OpConversionPattern<BufferOp> {
  AIELoweringPatternSets *lowering;
  using OpAdaptor = typename BufferOp::Adaptor;

public:
  BufferOpToAieLowering(mlir::MLIRContext *context,
                        AIELoweringPatternSets *lowering)
      : OpConversionPattern<BufferOp>(context), lowering(lowering) {}

  mlir::LogicalResult
  matchAndRewrite(BufferOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (auto address = op.getOperation()->getAttrOfType<StringAttr>(
            "aie.external_address")) {
      rewriter.replaceOpWithNewOp<xilinx::AIE::ExternalBufferOp>(
          op, op.getBuffer().getType(), lexicalCast<long long>(address.str()));

    } else {
      auto tile = lowering->getTile(op);
      rewriter.replaceOpWithNewOp<xilinx::AIE::BufferOp>(
          op, op.getBuffer().getType(), tile);
    }

    return success();
  }
};

void BufferOpLoweringPatternSet::populatePatternSet(
    mlir::RewritePatternSet &patterns) {

  patterns.add<BufferOpToAieLowering>(patterns.getContext(), lowering);
}
