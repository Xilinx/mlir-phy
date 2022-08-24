//===- StreamHubOp.cpp ------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
