//===- TargetResources.cpp --------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "phy/Target/AIE/TargetResources.h"
#include "phy/Dialect/Spatial/SpatialDialect.h"

#include <utility>

using namespace phy::connectivity;
using namespace phy::spatial;
using namespace phy::target;
using namespace phy::target::aie;

static std::string tileString(int col, int row) {
  return std::to_string(col) + "." + std::to_string(row);
}

std::list<VirtualResource>
TargetResources::VirtualResourceVertices(std::string virt_key) {
  std::list<VirtualResource> results;

  if (virt_key == "core" || virt_key == "locked_buffer") {
    for (int col = 1; col <= array_width; col++) {
      for (int row = 1; row <= array_height; row++) {
        // "tile/1.1/core" or "tile/1.1/locked_buffer"
        results.push_back(
            VirtualResource(virt_key, {{"tile", tileString(col, row)}}));
      }
    }
  }

  return results;
}

std::list<VirtualResource>
TargetResources::VirtualResourceNeighbors(VirtualResource &slot) {
  return std::list<VirtualResource>({});
};

Capacity TargetResources::VirtualResourceCapacity(VirtualResource &virt) {
  return TargetResourcesBase::VirtualResourceCapacity(virt);
};

TargetSupport TargetResources::PhysicalResourceSupport(PhysicalResource &phy) {
  if (physical_support.count(phy.key)) {
    return physical_support[phy.key];
  } else {
    return TargetResourcesBase::PhysicalResourceSupport(phy);
  }
};

Capacity TargetResources::PhysicalResourceCapacity(PhysicalResource &phy) {
  if (phy.key == "stream") {
    return stream_port_capacity[phy.metadata["port"]];
  }

  if (physical_capacity.count(phy.key)) {
    return physical_capacity[phy.key];
  } else {
    return TargetResourcesBase::PhysicalResourceCapacity(phy);
  }
};

Utilization
TargetResources::PhysicalResourceUtilization(PhysicalResource &phy,
                                             mlir::Operation *vertex) {
  return TargetResourcesBase::PhysicalResourceUtilization(phy, vertex);
};
