//===- StreamHub.h ----------------------------------------------*- C++ -*-===//
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_CONNECTIVITY_IMPLEMENTATION_STREAM_HUB_H
#define MLIR_PHY_CONNECTIVITY_IMPLEMENTATION_STREAM_HUB_H

#include "phy/Connectivity/Implementation.h"
#include "phy/Dialect/Physical/PhysicalDialect.h"
#include "phy/Dialect/Spatial/SpatialDialect.h"

#include <list>

namespace phy {
namespace connectivity {

class StreamHubImplementation : public Implementation {

  // Overrides
protected:
  mlir::Operation *createOperation() override;

public:
  using Implementation::Implementation;
  ~StreamHubImplementation() override {}

  void addPredecessor(std::weak_ptr<Implementation> pred, mlir::Operation *src,
                      mlir::Operation *dest) override;
  void addSuccessor(std::weak_ptr<Implementation> succ, mlir::Operation *src,
                    mlir::Operation *dest) override;

protected:
  std::list<std::weak_ptr<Implementation>> preds;
  std::list<std::weak_ptr<Implementation>> succs;
};

} // namespace connectivity
} // namespace phy

#endif // MLIR_PHY_CONNECTIVITY_IMPLEMENTATION_STREAM_HUB_H
