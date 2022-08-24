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

  // tiles[{col, row}] == TileOp
  std::map<std::pair<int, int>, xilinx::AIE::TileOp> tiles;

  // dmas/shim_dmas[{col, row}] == DMAOp/ShimDMAOp
  std::map<std::pair<int, int>, xilinx::AIE::MemOp> dmas;
  std::map<std::pair<int, int>, xilinx::AIE::ShimDMAOp> shim_dmas;

  std::map<phy::physical::BufferOp, xilinx::AIE::BufferOp> buffers;
  std::map<phy::physical::BufferOp, xilinx::AIE::ExternalBufferOp>
      external_buffers;
  std::map<phy::physical::CoreOp, xilinx::AIE::CoreOp> cores;
  std::map<phy::physical::LockOp, xilinx::AIE::LockOp> locks;

  xilinx::AIE::MemOp getDma(std::pair<int, int> index);
  xilinx::AIE::ShimDMAOp getShimDma(std::pair<int, int> index);

  xilinx::AIE::DMAChan getChannel(mlir::OpState &op);
  int getId(mlir::OpState &op);
  std::string getImpl(mlir::OpState &op);
  xilinx::AIE::WireBundle getWireBundle(phy::physical::StreamOp &op);

  xilinx::AIE::TileOp getTile(mlir::OpState &op);
  xilinx::AIE::TileOp getTile(std::pair<int, int> index);
  std::pair<int, int> getTileIndex(mlir::OpState &op);
};

} // namespace aie
} // namespace target
} // namespace phy

#endif // MLIR_PHY_TARGET_AIE_TARGET_LOWERING_PATTERNS_H
