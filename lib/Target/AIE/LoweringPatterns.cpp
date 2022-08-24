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

#include <map>
#include <utility>

#include "aie/AIEDialect.h"

using namespace mlir;
using namespace phy::connectivity;
using namespace phy::target;
using namespace phy::target::aie;
using namespace std;
using namespace xilinx;

list<unique_ptr<LoweringPatternSet>> AIELoweringPatternSets::getPatternSets() {
  list<unique_ptr<LoweringPatternSet>> patterns;

  patterns.push_back(make_unique<BufferOpLoweringPatternSet>(this));
  patterns.push_back(make_unique<LockOpLoweringPatternSet>(this));
  patterns.push_back(make_unique<StreamOpLoweringPatternSet>(this));

  patterns.push_back(make_unique<CoreOpLoweringPatternSet>(this));
  patterns.push_back(make_unique<StreamDmaOpLoweringPatternSet>(this));

  return patterns;
}

AIE::TileOp AIELoweringPatternSets::getTile(mlir::OpState &op) {
  return getTile(getTileIndex(op));
}

pair<int, int> AIELoweringPatternSets::getTileIndex(mlir::OpState &op) {
  auto tile = LiteralVector<int>(
      op.getOperation()->getAttrOfType<StringAttr>("aie.tile").str());
  return make_pair(tile.vec()[0], tile.vec()[1]);
}

AIE::TileOp AIELoweringPatternSets::getTile(pair<int, int> index) {
  if (!tiles.count(index)) {
    auto builder = OpBuilder::atBlockBegin(module.getBody());
    tiles[index] = builder.create<AIE::TileOp>(builder.getUnknownLoc(),
                                               index.first, index.second);
  }

  return tiles[index];
}

int AIELoweringPatternSets::getId(mlir::OpState &op) {
  return lexical_cast<int>(
      op.getOperation()->getAttrOfType<StringAttr>("aie.id").str());
}

AIE::DMAChan AIELoweringPatternSets::getChannel(mlir::OpState &op) {
  map<pair<string, int>, AIE::DMAChan> channels = {
      {{"S2MM", 0}, AIE::DMAChan::S2MM0},
      {{"S2MM", 1}, AIE::DMAChan::S2MM1},
      {{"MM2S", 0}, AIE::DMAChan::MM2S0},
      {{"MM2S", 1}, AIE::DMAChan::MM2S1}};

  auto engine =
      op.getOperation()->getAttrOfType<StringAttr>("aie.engine").str();
  auto id = getId(op);
  auto pair = make_pair(engine, id);

  assert(channels.count(pair) && "unknown engine");
  return channels[pair];
}

template <typename DMAOp>
static DMAOp getDmaGeneric(pair<int, int> index, mlir::ModuleOp module,
                           AIE::TileOp tile, map<pair<int, int>, DMAOp> &dmas) {

  if (!dmas.count(index)) {
    auto builder = OpBuilder::atBlockEnd(module.getBody());
    auto dma = dmas[index] = builder.create<DMAOp>(
        builder.getUnknownLoc(), builder.getIndexType(), tile);

    builder = OpBuilder::atBlockEnd(&dma.body().emplaceBlock());
    builder.create<AIE::EndOp>(builder.getUnknownLoc());
  }

  return dmas[index];
}

AIE::MemOp AIELoweringPatternSets::getDma(pair<int, int> index) {
  assert(!TargetResources().isShimTile(index.first, index.second));
  return getDmaGeneric<AIE::MemOp>(index, module, getTile(index), dmas);
}

AIE::ShimDMAOp AIELoweringPatternSets::getShimDma(pair<int, int> index) {
  assert(TargetResources().isShimTile(index.first, index.second));
  return getDmaGeneric<AIE::ShimDMAOp>(index, module, getTile(index),
                                       shim_dmas);
}
