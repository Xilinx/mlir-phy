//===- TargetResources.h ----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_TARGET_AIE_TARGET_RESOURCES_H
#define MLIR_PHY_TARGET_AIE_TARGET_RESOURCES_H

#include "phy/Target/Base/TargetResources.h"

#include <set>

#include "mlir/Support/LLVM.h"

namespace phy {
namespace target {
namespace aie {

using namespace phy::connectivity;

class TargetResources : phy::target::TargetResourcesBase {
  int array_height = 8;
  int array_width = 50;

  std::map<std::string, TargetSupport> physical_support = {
      {"lock", {{"states", 2}}},
      {"stream", {{"width_bytes", 32}}},
  };

  std::map<std::string, Capacity> physical_capacity = {
      {"buffer", {{"depth_bytes", 32 * 1024}}},
      {"core", {{"count", 1}, {"depth_bytes", 16 * 1024}}},
      {"lock", {{"count", 16}}},
      {"stream_dma", {{"count", 2}}}};

  std::map<std::string, Capacity> stream_port_capacity = {
      {"Core.I", {{"count", 2}}},
      {"Core.O", {{"count", 2}}},
      {"DMA.I", {{"count", 2}}},
      {"DMA.O", {{"count", 2}}},
      {"FIFO.I", {{"count", 2}}},
      {"FIFO.O", {{"count", 2}}},
      {"North.I", {{"count", 4}}},
      {"North.O", {{"count", 6}}},
      // South.O is the same endpoint of North.I of its south switch
      {"East.I", {{"count", 4}}},
      {"East.O", {{"count", 4}}},
      // West.O is the same endpoint of East.I of its south switch
  };

  bool isLegalAffinity(int coreCol, int coreRow, int bufCol, int bufRow);
  std::set<std::pair<int, int>> getAffinity(int col, int row,
                                            std::string neigh_type);

public:
  std::list<VirtualResource>
  VirtualResourceVertices(std::string virt_key) override;
  std::list<VirtualResource>
  VirtualResourceNeighbors(VirtualResource &slot) override;
  Capacity VirtualResourceCapacity(VirtualResource &virt) override;

  TargetSupport PhysicalResourceSupport(PhysicalResource &phy) override;
  Capacity PhysicalResourceCapacity(PhysicalResource &phy) override;
  Utilization PhysicalResourceUtilization(PhysicalResource &phy,
                                          mlir::Operation *vertex) override;
};

} // namespace aie
} // namespace target
} // namespace phy

#endif // MLIR_PHY_TARGET_AIE_TARGET_RESOURCES_H
