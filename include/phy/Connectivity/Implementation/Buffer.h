//===- Buffer.h -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_CONNECTIVITY_IMPLEMENTATION_BUFFER_H
#define MLIR_PHY_CONNECTIVITY_IMPLEMENTATION_BUFFER_H

#include "phy/Connectivity/Implementation.h"
#include "phy/Dialect/Physical/PhysicalDialect.h"
#include "phy/Dialect/Spatial/SpatialDialect.h"

namespace phy {
namespace connectivity {

class BufferImplementation : public Implementation {

  // Overrides
protected:
  mlir::Operation *createOperation() override;

public:
  using Implementation::Implementation;
  void addSpatialOperation(mlir::Operation *spatial) override;
  void translateUserOperation(mlir::Value value,
                              mlir::Operation *user) override;

protected:
  spatial::QueueOp queue;
};

} // namespace connectivity
} // namespace phy

#endif // MLIR_PHY_CONNECTIVITY_IMPLEMENTATION_BUFFER_H
