//===- LayoutToPhysical.cpp -----------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "phy/Conversion/LayoutToPhysical.h"

#include "phy/Connectivity/Implementation.h"
#include "phy/Connectivity/ResourceList.h"
#include "phy/Conversion/Passes.h"
#include "phy/Dialect/Layout/LayoutDialect.h"
#include "phy/Dialect/Physical/PhysicalDialect.h"
#include "phy/Dialect/Spatial/SpatialDialect.h"
#include "phy/Rewrite/RemoveOp.h"

#include <memory>
#include <set>

#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "convert-layout-to-physical"

using namespace mlir;
using namespace phy;
using namespace phy::rewrite;
using namespace phy::connectivity;

namespace {

struct LayoutToPhysical : public LayoutToPhysicalBase<LayoutToPhysical> {

  void runOnOperation() override {
    ImplementationContext context(getOperation(), device_option);
    collectImplementations(context);
    populateNeighborInfo(context);
    getOperationForImplementations(context);
    cleanupSpatialLayoutOperations(context);
  }

  void collectImplementations(ImplementationContext &context) {
    context.module.walk([&](Operation *Op) {
      auto device = dyn_cast<layout::DeviceOp>(Op);

      // skipping non device operations
      // TODO: support platforms
      if (!device)
        return;

      // skipping devices that are not selected.
      if (device.device().str() != device_option)
        return;

      // TODO: support route
      for (auto place : device.getOps<layout::PlaceOp>()) {
        ResourceList resources(place.slot().str());
        auto spatial_op = place.vertex().getDefiningOp();

        for (auto phy : resources.phys) {
          auto impl = context.getImplementation(phy);
          if (impl) {
            impl->addSpatialOperation(spatial_op);
          } else {
            place->emitWarning() << phy.key << " cannot be implemented.";
          }
        }
      }
    });
  }

  void populateNeighborInfo(ImplementationContext &context) {}

  void getOperationForImplementations(ImplementationContext &context) {
    // Making sure each is implemented as an operations
    for (auto impl : context.impls) {
      if (impl.second) {
        impl.second->getOperation();
      }
    }
  }

  void cleanupSpatialLayoutOperations(ImplementationContext &context) {
    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<physical::PhysicalDialect>();
    target.addIllegalDialect<layout::LayoutDialect>();
    target.addIllegalDialect<spatial::SpatialDialect>();

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<OpRemover<layout::PlatformOp>>(&getContext());
    patterns.add<OpRemover<layout::DeviceOp>>(&getContext());
    patterns.add<OpRemover<layout::PlaceOp>>(&getContext());
    patterns.add<OpRemover<layout::RouteOp>>(&getContext());
    patterns.add<OpRemover<spatial::QueueOp>>(&getContext());
    patterns.add<OpRemover<spatial::NodeOp>>(&getContext());

    if (mlir::failed(mlir::applyPartialConversion(context.module, target,
                                                  std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> phy::createLayoutToPhysical() {
  return std::make_unique<LayoutToPhysical>();
}
