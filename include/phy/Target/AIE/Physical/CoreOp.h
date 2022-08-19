//===- CoreOp.h -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "phy/Target/Base/LoweringPatterns.h"

#include "mlir/Transforms/DialectConversion.h"

#ifndef MLIR_PHY_TARGET_AIE_TARGET_PHYSICAL_COREOP_H
#define MLIR_PHY_TARGET_AIE_TARGET_PHYSICAL_COREOP_H

namespace phy {
namespace target {
namespace aie {

class CoreOpLoweringPatternSet : public LoweringPatternSet {
public:
  void populatePatternSet(mlir::RewritePatternSet &patterns,
                          mlir::MLIRContext *context) override;
};

} // namespace aie
} // namespace target
} // namespace phy

#endif // MLIR_PHY_TARGET_AIE_TARGET_PHYSICAL_COREOP_H
