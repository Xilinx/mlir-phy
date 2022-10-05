//===- LexicalCast.h --------------------------------------------*- C++ -*-===//
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "phy/Target/AIE/LoweringPatterns.h"
#include "phy/Target/Base/LoweringPatterns.h"

#include "mlir/Transforms/DialectConversion.h"

#ifndef MLIR_PHY_SUPPORT_LEXICAL_CAST_H
#define MLIR_PHY_SUPPORT_LEXICAL_CAST_H

template <typename T2, typename T1>
inline T2 lexicalCast(const T1 &in) {
  T2 out;
  std::stringstream ss;
  ss << in;
  ss >> out;
  return out;
}

#endif // MLIR_PHY_SUPPORT_LEXICAL_CAST_H
