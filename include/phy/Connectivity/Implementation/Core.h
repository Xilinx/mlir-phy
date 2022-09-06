//===- Core.h ---------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_CONNECTIVITY_IMPLEMENTATION_CORE_H
#define MLIR_PHY_CONNECTIVITY_IMPLEMENTATION_CORE_H

#include "phy/Connectivity/Implementation.h"
#include "phy/Dialect/Physical/PhysicalDialect.h"
#include "phy/Dialect/Spatial/SpatialDialect.h"

#include <list>
#include <map>
#include <memory>

namespace phy {
namespace connectivity {

class CoreImplementation : public Implementation {
  spatial::NodeOp node;
  std::map<spatial::QueueOp, std::list<std::weak_ptr<Implementation>>>
      queue_impls;

  void addQueueImpl(spatial::QueueOp queue, std::weak_ptr<Implementation> impl);
  llvm::SmallVector<mlir::Value> getOperandValues(mlir::Value operand);
  mlir::StringRef translateFunction();
  llvm::SmallVector<mlir::Value> translateOperands();

public:
  using Implementation::Implementation;
  mlir::Operation *createOperation() override;
  void addPredecessor(std::weak_ptr<Implementation> pred, mlir::Operation *src,
                      mlir::Operation *dest) override;
  void addSuccessor(std::weak_ptr<Implementation> succ, mlir::Operation *src,
                    mlir::Operation *dest) override;
  void addSpatialOperation(mlir::Operation *spatial) override;
};

} // namespace connectivity
} // namespace phy

#endif // MLIR_PHY_CONNECTIVITY_IMPLEMENTATION_CORE_H
