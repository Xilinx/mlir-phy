//===- StreamHubOp.cpp ------------------------------------------*- C++ -*-===//
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "phy/Target/AIE/Physical/StreamHubOp.h"

#include "mlir/Transforms/DialectConversion.h"
#include "phy/Target/AIE/Physical/Implementation/BroadcastPacket.h"

using namespace mlir;
using namespace phy::target::aie;

void StreamHubOpLoweringPatternSet::populatePatternSet(
    mlir::RewritePatternSet &patterns) {

  patterns.add<BroadcastPacketLowering>(patterns.getContext(), lowering);
}
