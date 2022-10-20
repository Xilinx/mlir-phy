//===- StreamHubOp.h ---------------------------------------------*- C++-*-===//
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "phy/Target/AIE/LoweringPatterns.h"
#include "phy/Target/Base/LoweringPatterns.h"

#include "mlir/Transforms/DialectConversion.h"

#ifndef MLIR_PHY_TARGET_AIE_TARGET_PHYSICAL_STREAM_HUB_OP_H
#define MLIR_PHY_TARGET_AIE_TARGET_PHYSICAL_STREAM_HUB_OP_H

namespace phy {
namespace target {
namespace aie {

class StreamHubOpLoweringPatternSet : public LoweringPatternSet {
  AIELoweringPatternSets *lowering;

public:
  StreamHubOpLoweringPatternSet(AIELoweringPatternSets *lowering)
      : lowering(lowering){};
  ~StreamHubOpLoweringPatternSet() override {}

  void populatePatternSet(mlir::RewritePatternSet &patterns) override;
};

} // namespace aie
} // namespace target
} // namespace phy

#endif // MLIR_PHY_TARGET_AIE_TARGET_PHYSICAL_STREAM_HUB_OP_H
