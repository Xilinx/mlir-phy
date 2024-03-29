//===- TargetResources.h ----------------------------------------*- C++ -*-===//
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_TARGET_BASE_TARGET_RESOURCES_H
#define MLIR_PHY_TARGET_BASE_TARGET_RESOURCES_H

#include "phy/Connectivity/Constraint.h"
#include "phy/Connectivity/Resource.h"

#include <list>

#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

namespace phy {
namespace target {

using namespace phy::connectivity;

class TargetResourcesBase {

public:
  /**
   * Connectivity graph of virtual resources.
   */

  virtual std::list<VirtualResource>
  getVirtualResourceVertices(std::string virt_key) {
    return std::list<VirtualResource>();
  }

  virtual std::list<VirtualResource>
  getVirtualResourceNeighbors(VirtualResource &slot) {
    return std::list<VirtualResource>();
  }

  virtual Capacity getVirtualResourceCapacity(VirtualResource &virt) {
    return Capacity({
        {"count", 1},
    });
  }

  virtual Utilization getVirtualResourceUtilization(VirtualResource &virt,
                                                    mlir::Operation *vertex) {
    return Utilization({
        {"count", 1},
    });
  }

  /**
   * Target support of physical resources.
   */

  virtual TargetSupport getPhysicalResourceSupport(PhysicalResource &phy) {
    return TargetSupport({});
  }

  virtual Capacity getPhysicalResourceCapacity(PhysicalResource &phy) {
    return Capacity({
        {"count", 1},
    });
  }

  virtual Utilization getPhysicalResourceUtilization(PhysicalResource &phy,
                                                     mlir::Operation *vertex) {
    return Utilization({
        {"count", 1},
    });
  }

  virtual ~TargetResourcesBase() {}
};

} // namespace target
} // namespace phy

#endif // MLIR_PHY_TARGET_BASE_TARGET_RESOURCES_H
