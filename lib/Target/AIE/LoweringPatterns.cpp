//===- LoweringPatterns.cpp -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "phy/Target/AIE/LoweringPatterns.h"
#include "phy/Target/AIE/Physical/CoreOp.h"

using namespace phy::target;
using namespace phy::target::aie;

std::list<std::unique_ptr<LoweringPatternSet>>
AIELoweringPatternSets::getPatternSets() {
  std::list<std::unique_ptr<LoweringPatternSet>> patterns;

  patterns.push_back(std::make_unique<CoreOpLoweringPatternSet>());

  return patterns;
}