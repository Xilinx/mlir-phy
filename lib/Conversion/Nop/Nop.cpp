//===- Nop.cpp ------------------------------------------------------------===//
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
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
