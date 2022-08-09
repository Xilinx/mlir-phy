//===- PhyToAie.cpp -------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "phy/Conversion/PhysicalToAie.h"
#include "phy/Conversion/Passes.h"
#include "phy/Dialect/Physical/PhysicalDialect.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "convert-physical-to-aie"

using namespace mlir;
using namespace phy;

namespace {

struct PhysicalToAie : public PhysicalToAieBase<PhysicalToAie> {
  void runOnOperation() override {}
};

} // namespace

std::unique_ptr<mlir::Pass> phy::createPhysicalToAie() {
  return std::make_unique<PhysicalToAie>();
}
