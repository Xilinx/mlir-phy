//===- LoweringPatterns.h ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
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
  ModuleOp module;

  // dmas/shim_dmas[{col, row}] == DMAOp/ShimDMAOp
  std::map<std::pair<int, int>, xilinx::AIE::MemOp> dmas;
  std::map<std::pair<int, int>, xilinx::AIE::ShimDMAOp> shim_dmas;

  // locks[{TileOp, id}] == LockOp
  std::map<std::pair<xilinx::AIE::TileOp, int>, xilinx::AIE::LockOp> locks;

  // tiles[{col, row}] == TileOp
  std::map<std::pair<int, int>, xilinx::AIE::TileOp> tiles;

public:
  AIELoweringPatternSets(ModuleOp &module) : module(module) {}
  ~AIELoweringPatternSets() override {}

  // Lists of lowering pattern sets
  std::list<std::list<std::unique_ptr<LoweringPatternSet>>>
  getPatternSets() override;

  // Shared resources constructors and getters.
  xilinx::AIE::MemOp getDma(std::pair<int, int> index);
  xilinx::AIE::LockOp getLock(xilinx::AIE::TileOp tile, int id);
  xilinx::AIE::ShimDMAOp getShimDma(std::pair<int, int> index);
  xilinx::AIE::TileOp getTile(mlir::OpState &op);
  xilinx::AIE::TileOp getTile(std::pair<int, int> index);

  // Common attribute getters.
  xilinx::AIE::DMAChan getChannel(mlir::OpState &op,
                                  phy::physical::StreamOp stream);
  int getId(mlir::OpState &op);
  std::string getImpl(mlir::OpState &op);
  std::pair<int, int> getTileIndex(mlir::OpState &op);
  xilinx::AIE::WireBundle getWireBundle(phy::physical::StreamOp &op);
};

} // namespace aie
} // namespace target
} // namespace phy

#endif // MLIR_PHY_TARGET_AIE_TARGET_LOWERING_PATTERNS_H
