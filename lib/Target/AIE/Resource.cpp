//===- Resource.cpp ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "phy/Target/AIE/Resource.h"
#include "phy/Dialect/Spatial/SpatialDialect.h"

#include <utility>

using namespace phy::connectivity;
using namespace phy::spatial;
using namespace phy::target;
using namespace phy::target::aie;

static std::string tileString(int col, int row) {
  return std::to_string(col) + "." + std::to_string(row);
}

static std::pair<int, int> parseTile(std::string tile) {
  size_t pos = tile.find(".");

  int col = std::stoi(tile.substr(0, pos));
  tile.erase(0, pos + 1);
  int row = std::stoi(tile);

  return {col, row};
}

std::list<Phy> Resource::getDummyPhys(std::set<std::pair<int, int>> tiles,
                                      std::string phy_key) {
  std::list<Phy> phys;
  for (auto tile : tiles) {
    phys.push_back(
        Phy(phy_key, {{"tile", tileString(tile.first, tile.second)}}));
  }
  return phys;
}

Capacity Resource::PhyCapacity(Phy &phy) {
  if (phy.key == "stream") {
    return stream_port_capacity[phy.metadata["port"]];
  }

  if (method_capacity.count(phy.key)) {
    return method_capacity[phy.key];
  } else {
    return ResourceBase::PhyCapacity(phy);
  }
}

TargetSupport Resource::PhyTargetSupport(Phy &phy) {
  if (method_support.count(phy.key)) {
    return method_support[phy.key];
  } else {
    return ResourceBase::PhyTargetSupport(phy);
  }
}

std::list<Slot> Resource::FittableSlots(mlir::Operation *vertex) {
  std::list<Slot> results;

  if (auto node = mlir::dyn_cast<NodeOp>(*vertex)) {

    for (int col = 1; col <= array_width; col++) {
      for (int row = 1; row <= array_height; row++) {
        // "tile/1.1/core,tile/2.1/buffer_affinity,..."
        auto phys =
            getDummyPhys(getAffinity(col, row, "buffer"), "buffer_affinity");
        phys.push_front(Phy("core", {{"tile", tileString(col, row)}}));
        results.push_back(Slot({phys}));
      }
    }

  } else if (auto queue = mlir::dyn_cast<QueueOp>(*vertex)) {

    for (int col = 1; col <= array_width; col++) {
      for (int row = 1; row <= array_height; row++) {
        // "tile/1.1/buffer,tile/2.1/core_affinity,..."
        auto phys =
            getDummyPhys(getAffinity(col, row, "core"), "core_affinity");
        phys.push_front(Phy("buffer", {{"tile", tileString(col, row)}}));
        results.push_back(Slot(phys));
      }
    }
  }

  return results;
}

std::list<Slot> Resource::SlotNeighbors(Slot &slot) {
  std::string main_key = slot.phys.front().key;

  std::list<Slot> neighbors;

  if (main_key == std::string("core")) {
    // 1. Connect to locks
    for (auto phy : slot.phys) {
      if (phy.key == "buffer_affinity") {
        auto tile = parseTile(phy.metadata["tile"]);
        int col = tile.first, row = tile.second;

        // "tile/1.1/lock,tile/2.1/buffer_affinity,tile/2.1/core_affinity,..."
        auto phys =
            getDummyPhys(getAffinity(col, row, "core"), "core_affinity");
        phys.push_front(Phy("lock", {{"tile", tileString(col, row)}}));
        neighbors.push_back(Slot(phys));
      }
    }
  }

  if (main_key == std::string("buffer")) {
  }

  if (main_key == std::string("lock")) {
  }

  return neighbors;
}
