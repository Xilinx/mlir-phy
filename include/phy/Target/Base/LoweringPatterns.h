//===- LoweringPatterns.h ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
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

  virtual ~LoweringPatternSet() {}
};

class LoweringPatternSets {
public:
  virtual std::list<std::list<std::unique_ptr<LoweringPatternSet>>>
  getPatternSets() {
    return std::list<std::list<std::unique_ptr<LoweringPatternSet>>>();
  }

  virtual ~LoweringPatternSets() {}
};

} // namespace target
} // namespace phy

#endif // MLIR_PHY_TARGET_BASE_LOWERING_PATTERNS_H
