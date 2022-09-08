//===- StreamDmaOp.h ---------------------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "phy/Target/AIE/LoweringPatterns.h"
#include "phy/Target/Base/LoweringPatterns.h"

#include "mlir/Transforms/DialectConversion.h"

#ifndef MLIR_PHY_TARGET_AIE_TARGET_PHYSICAL_STREAM_DMA_OP_H
#define MLIR_PHY_TARGET_AIE_TARGET_PHYSICAL_STREAM_DMA_OP_H

namespace phy {
namespace target {
namespace aie {

class StreamDmaOpLoweringPatternSet : public LoweringPatternSet {
  AIELoweringPatternSets *lowering;

public:
  StreamDmaOpLoweringPatternSet(AIELoweringPatternSets *lowering)
      : lowering(lowering){};
  ~StreamDmaOpLoweringPatternSet() override {}

  void populatePatternSet(mlir::RewritePatternSet &patterns) override;
  void populateTarget(mlir::ConversionTarget &target) override;
};

} // namespace aie
} // namespace target
} // namespace phy

#endif // MLIR_PHY_TARGET_AIE_TARGET_PHYSICAL_STREAM_DMA_OP_H
