//===- Lock.h ---------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_CONNECTIVITY_IMPLEMENTATION_LOCK_H
#define MLIR_PHY_CONNECTIVITY_IMPLEMENTATION_LOCK_H

#include "phy/Connectivity/Implementation.h"
#include "phy/Dialect/Physical/PhysicalDialect.h"
#include "phy/Dialect/Spatial/SpatialDialect.h"

namespace phy {
namespace connectivity {

class LockImplementation : public Implementation {

  // Overrides
protected:
  mlir::Operation *createOperation() override;

public:
  using Implementation::Implementation;
  ~LockImplementation() override {}

  void translateUserOperation(mlir::Value value,
                              mlir::Operation *user) override;
};

} // namespace connectivity
} // namespace phy

#endif // MLIR_PHY_CONNECTIVITY_IMPLEMENTATION_LOCK_H
