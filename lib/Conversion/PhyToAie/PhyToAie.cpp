//===- PhyToAie.cpp -------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "phy/Conversion/PhyToAie.h"
#include "phy/Conversion/Passes.h"
#include "phy/Dialect/Phy/PhyDialect.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "convert-phy-to-aie"

using namespace mlir;
using namespace phy;

namespace {

struct PhyToAie : public PhyToAieBase<PhyToAie> {
  void runOnOperation() override {}
};

} // namespace

std::unique_ptr<mlir::Pass> phy::createPhyToAie() {
  return std::make_unique<PhyToAie>();
}
