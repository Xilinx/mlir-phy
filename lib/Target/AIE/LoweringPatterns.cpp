//===- LoweringPatterns.cpp -------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "phy/Target/AIE/LoweringPatterns.h"

#include "phy/Connectivity/Serialization/LiteralVector.h"
#include "phy/Target/AIE/Physical/BufferOp.h"
#include "phy/Target/AIE/Physical/CoreOp.h"

#include "aie/AIEDialect.h"

using namespace mlir;
using namespace phy::connectivity;
using namespace phy::target;
using namespace phy::target::aie;

std::list<std::unique_ptr<LoweringPatternSet>>
AIELoweringPatternSets::getPatternSets() {
  std::list<std::unique_ptr<LoweringPatternSet>> patterns;

  patterns.push_back(std::make_unique<BufferOpLoweringPatternSet>(this));
  patterns.push_back(std::make_unique<CoreOpLoweringPatternSet>(this));

  return patterns;
}

xilinx::AIE::TileOp AIELoweringPatternSets::getTile(mlir::OpState &op) {
  return getTileOp(getTileAttr(op));
}

std::pair<int, int> AIELoweringPatternSets::getTileAttr(mlir::OpState &op) {
  auto tile = LiteralVector<int>(
      op.getOperation()->getAttrOfType<StringAttr>("aie.tile").str());
  return std::make_pair(tile.vec()[0], tile.vec()[1]);
}

xilinx::AIE::TileOp
AIELoweringPatternSets::getTileOp(std::pair<int, int> index) {
  if (!tiles.count(index)) {
    auto builder = OpBuilder::atBlockBegin(module.getBody());
    tiles[index] = builder.create<xilinx::AIE::TileOp>(
        builder.getUnknownLoc(), index.first, index.second);
  }

  return tiles[index];
}