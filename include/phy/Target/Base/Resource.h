//===- Resource.h -----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_TARGET_BASE_RESOURCE_H
#define MLIR_PHY_TARGET_BASE_RESOURCE_H

#include "phy/Connectivity/Constraint.h"
#include "phy/Connectivity/Graph.h"

#include <list>

#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

namespace phy {
namespace target {

using namespace phy::connectivity;

class ResourceBase {

public:
  virtual Capacity PhyCapacity(Phy &phy) {
    return Capacity({
        {"count", 1},
    });
  }

  virtual TargetSupport PhyTargetSupport(Phy &phy) { return TargetSupport({}); }

  virtual std::list<Slot> FittableSlots(mlir::Operation *vertex) {
    return std::list<Slot>();
  }

  virtual std::list<Slot> SlotNeighbors(Slot &slot) {
    return std::list<Slot>();
  }
};

} // namespace target
} // namespace phy

#endif // MLIR_PHY_TARGET_BASE_RESOURCE_H
