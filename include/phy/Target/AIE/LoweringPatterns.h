//===- LoweringPatterns.h ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "phy/Target/Base/LoweringPatterns.h"

#include <list>
#include <memory>

#ifndef MLIR_PHY_TARGET_AIE_TARGET_LOWERING_PATTERNS_H
#define MLIR_PHY_TARGET_AIE_TARGET_LOWERING_PATTERNS_H

namespace phy {
namespace target {
namespace aie {

class AIELoweringPatternSets : public LoweringPatternSets {
public:
  std::list<std::unique_ptr<LoweringPatternSet>> getPatternSets() override;
};

} // namespace aie
} // namespace target
} // namespace phy

#endif // MLIR_PHY_TARGET_AIE_TARGET_LOWERING_PATTERNS_H
