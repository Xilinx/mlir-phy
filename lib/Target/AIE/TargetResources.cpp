//===- TargetResources.cpp --------------------------------------*- C++ -*-===//
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
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
TargetResources::getVirtualResourceVertices(std::string virt_key) {
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
TargetResources::getVirtualResourceNeighbors(VirtualResource &slot) {
  return std::list<VirtualResource>({});
};

Capacity TargetResources::getVirtualResourceCapacity(VirtualResource &virt) {
  return TargetResourcesBase::getVirtualResourceCapacity(virt);
};

TargetSupport
TargetResources::getPhysicalResourceSupport(PhysicalResource &phy) {
  if (physical_support.count(phy.key)) {
    return physical_support[phy.key];
  }
  return TargetResourcesBase::getPhysicalResourceSupport(phy);
};

Capacity TargetResources::getPhysicalResourceCapacity(PhysicalResource &phy) {
  if (phy.key == "stream") {
    return stream_port_capacity[phy.metadata["port"]];
  }

  if (physical_capacity.count(phy.key)) {
    return physical_capacity[phy.key];
  }

  return TargetResourcesBase::getPhysicalResourceCapacity(phy);
};

Utilization
TargetResources::getPhysicalResourceUtilization(PhysicalResource &phy,
                                                mlir::Operation *vertex) {
  return TargetResourcesBase::getPhysicalResourceUtilization(phy, vertex);
};
