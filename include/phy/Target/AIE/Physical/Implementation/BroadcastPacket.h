//===- BroadcastPacket.h -----------------------------------------*- C++-*-===//
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "phy/Dialect/Physical/PhysicalDialect.h"

#include "phy/Target/AIE/LoweringPatterns.h"

#include "mlir/Transforms/DialectConversion.h"

#ifndef MLIR_PHY_TARGET_AIE_TARGET_PHYSICAL_IMPL_BROADCAST_PACKET_H
#define MLIR_PHY_TARGET_AIE_TARGET_PHYSICAL_IMPL_BROADCAST_PACKET_H

namespace phy {
namespace target {
namespace aie {

class BroadcastPacketLowering
    : public mlir::OpConversionPattern<physical::StreamHubOp> {
  AIELoweringPatternSets *lowering;
  using OpAdaptor = typename physical::StreamHubOp::Adaptor;

public:
  BroadcastPacketLowering(mlir::MLIRContext *context,
                          AIELoweringPatternSets *lowering)
      : OpConversionPattern<physical::StreamHubOp>(context),
        lowering(lowering) {}

  mlir::LogicalResult
  matchAndRewrite(physical::StreamHubOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override;

  void buildBroadcastPacket(mlir::OpBuilder &builder, physical::StreamHubOp op,
                            physical::StreamOp &src) const;
  void buildBpId(mlir::OpBuilder &builder, physical::StreamHubOp op,
                 physical::StreamOp &src, int tag) const;
};

} // namespace aie
} // namespace target
} // namespace phy

#endif // MLIR_PHY_TARGET_AIE_TARGET_PHYSICAL_IMPL_BROADCAST_PACKET_H
