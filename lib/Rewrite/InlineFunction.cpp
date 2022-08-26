//===- InlineFunction.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "phy/Rewrite/InlineFunction.h"

#include "phy/Dialect/Physical/PhysicalDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::func;
using namespace phy::target::aie;

LogicalResult
FunctionInliner::matchAndRewrite(CallOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  Inliner inliner(rewriter.getContext());

  auto sym = op.getCallableForCallee().dyn_cast<SymbolRefAttr>();
  if (!sym)
    return failure();

  auto func = dyn_cast_or_null<func::FuncOp>(
      SymbolTable::lookupNearestSymbolFrom(op, sym));
  if (!func)
    return failure();

  return inlineCall(inliner, op, func, func.getCallableRegion());
}
