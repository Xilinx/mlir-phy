//===- LoweringPatterns.h ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "phy/Dialect/Physical/PhysicalDialect.h"
#include "phy/Target/Base/LoweringPatterns.h"

#include "aie/AIEDialect.h"

#include <list>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#ifndef MLIR_PHY_TARGET_AIE_TARGET_LOWERING_PATTERNS_H
#define MLIR_PHY_TARGET_AIE_TARGET_LOWERING_PATTERNS_H

namespace phy {
namespace target {
namespace aie {

class AIELoweringPatternSets : public LoweringPatternSets {
public:
  AIELoweringPatternSets(ModuleOp &module) : module(module) {}
  std::list<std::unique_ptr<LoweringPatternSet>> getPatternSets() override;

  ModuleOp module;
  std::map<std::pair<int, int>, xilinx::AIE::TileOp> tiles;
  std::map<phy::physical::BufferOp, xilinx::AIE::BufferOp> buffers;
  std::map<phy::physical::CoreOp, xilinx::AIE::CoreOp> cores;
  std::map<phy::physical::LockOp, xilinx::AIE::LockOp> locks;

  xilinx::AIE::TileOp getTile(mlir::OpState &op);
  int getId(mlir::OpState &op);

  std::pair<int, int> getTileAttr(mlir::OpState &op);
  xilinx::AIE::TileOp getTileOp(std::pair<int, int> index);
};

} // namespace aie
} // namespace target
} // namespace phy

#endif // MLIR_PHY_TARGET_AIE_TARGET_LOWERING_PATTERNS_H
