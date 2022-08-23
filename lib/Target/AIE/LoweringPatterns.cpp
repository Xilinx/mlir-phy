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
#include "phy/Target/AIE/Physical/LockOp.h"
#include "phy/Target/AIE/Physical/StreamDmaOp.h"
#include "phy/Target/AIE/Physical/StreamOp.h"
#include "phy/Target/AIE/TargetResources.h"

#include "aie/AIEDialect.h"

using namespace mlir;
using namespace phy::connectivity;
using namespace phy::target;
using namespace phy::target::aie;

std::list<std::unique_ptr<LoweringPatternSet>>
AIELoweringPatternSets::getPatternSets() {
  std::list<std::unique_ptr<LoweringPatternSet>> patterns;

  patterns.push_back(std::make_unique<BufferOpLoweringPatternSet>(this));
  patterns.push_back(std::make_unique<LockOpLoweringPatternSet>(this));
  patterns.push_back(std::make_unique<StreamOpLoweringPatternSet>(this));

  patterns.push_back(std::make_unique<CoreOpLoweringPatternSet>(this));
  patterns.push_back(std::make_unique<StreamDmaOpLoweringPatternSet>(this));

  return patterns;
}

xilinx::AIE::TileOp AIELoweringPatternSets::getTile(mlir::OpState &op) {
  return getTile(getTileIndex(op));
}

std::pair<int, int> AIELoweringPatternSets::getTileIndex(mlir::OpState &op) {
  auto tile = LiteralVector<int>(
      op.getOperation()->getAttrOfType<StringAttr>("aie.tile").str());
  return std::make_pair(tile.vec()[0], tile.vec()[1]);
}

xilinx::AIE::TileOp AIELoweringPatternSets::getTile(std::pair<int, int> index) {
  if (!tiles.count(index)) {
    auto builder = OpBuilder::atBlockBegin(module.getBody());
    tiles[index] = builder.create<xilinx::AIE::TileOp>(
        builder.getUnknownLoc(), index.first, index.second);
  }

  return tiles[index];
}

int AIELoweringPatternSets::getId(mlir::OpState &op) {
  return lexical_cast<int>(
      op.getOperation()->getAttrOfType<StringAttr>("aie.id").str());
}

template <typename DMAOp>
static DMAOp getDmaGeneric(std::pair<int, int> index, mlir::ModuleOp module,
                           xilinx::AIE::TileOp tile,
                           std::map<std::pair<int, int>, DMAOp> &dmas) {

  if (!dmas.count(index)) {
    auto builder = OpBuilder::atBlockEnd(module.getBody());
    auto dma = dmas[index] = builder.create<DMAOp>(
        builder.getUnknownLoc(), builder.getIndexType(), tile);

    builder = OpBuilder::atBlockEnd(&dma.body().emplaceBlock());
    builder.create<xilinx::AIE::EndOp>(builder.getUnknownLoc());
  }

  return dmas[index];
}

xilinx::AIE::MemOp AIELoweringPatternSets::getDma(std::pair<int, int> index) {
  assert(!TargetResources().isShimTile(index.first, index.second));
  return getDmaGeneric<xilinx::AIE::MemOp>(index, module, getTile(index), dmas);
}

xilinx::AIE::ShimDMAOp
AIELoweringPatternSets::getShimDma(std::pair<int, int> index) {
  assert(TargetResources().isShimTile(index.first, index.second));
  return getDmaGeneric<xilinx::AIE::ShimDMAOp>(index, module, getTile(index),
                                               shim_dmas);
}
