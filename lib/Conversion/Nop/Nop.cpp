//===- Nop.cpp ------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "phy/Conversion/Nop.h"
#include "phy/Conversion/Passes.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "convert-nop"

using namespace mlir;
using namespace phy;

namespace {

struct Nop : public NopBase<Nop> {
  void runOnOperation() override {}
};

} // namespace

std::unique_ptr<mlir::Pass> phy::createNop() { return std::make_unique<Nop>(); }