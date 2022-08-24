//===- PassDetail.h - Conversion Pass class details -------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CONVERSION_PASSDETAIL_H
#define CONVERSION_PASSDETAIL_H

#ifdef BUILD_WITH_AIE
#include "aie/AIEDialect.h"
#endif

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

#include "phy/Conversion/Nop.h"
#include "phy/Conversion/PhysicalToAie.h"
#include "phy/Dialect/Layout/LayoutDialect.h"
#include "phy/Dialect/Physical/PhysicalDialect.h"
#include "phy/Dialect/Spatial/SpatialDialect.h"

namespace phy {
    
// Generate the classes which represent the passes
#define GEN_PASS_CLASSES
#define GEN_PASS_REGISTRATION
#include "phy/Conversion/Passes.h.inc"

}

#endif // CONVERSION_PASSDETAIL_H
