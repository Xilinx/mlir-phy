//===- LoweringPatterns.h ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <list>

#include "mlir/IR/Dialect.h"
#include "mlir/Transforms/DialectConversion.h"

#ifndef MLIR_PHY_TARGET_BASE_LOWERING_PATTERNS_H
#define MLIR_PHY_TARGET_BASE_LOWERING_PATTERNS_H

namespace phy {
namespace target {

class LoweringPatternSet {
public:
  virtual void populatePatternSet(mlir::RewritePatternSet &patterns) {}
  virtual void populateTarget(mlir::ConversionTarget &target) {}
};

class LoweringPatternSets {
public:
  virtual std::list<std::list<std::unique_ptr<LoweringPatternSet>>>
  getPatternSets() {
    return std::list<std::list<std::unique_ptr<LoweringPatternSet>>>();
  }
};

} // namespace target
} // namespace phy

#endif // MLIR_PHY_TARGET_BASE_LOWERING_PATTERNS_H
