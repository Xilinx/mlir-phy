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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"

#include "phy/Conversion/PhyToAie.h"
#include "phy/Dialect/Phy/PhyDialect.h"

namespace phy {
    
// Generate the classes which represent the passes
#define GEN_PASS_CLASSES
#define GEN_PASS_REGISTRATION
#include "phy/Conversion/Passes.h.inc"

}

#endif // CONVERSION_PASSDETAIL_H
