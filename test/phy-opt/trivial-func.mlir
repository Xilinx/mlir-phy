// RUN: phy-opt %s | FileCheck %s
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @function() -> i1
func.func @function() -> i1 {
  %0 = llvm.mlir.constant(false) : i1
  func.return %0 : i1
}