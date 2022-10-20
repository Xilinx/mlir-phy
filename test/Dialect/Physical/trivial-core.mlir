// RUN: phy-opt %s | FileCheck %s
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @function()
func.func @function() {
  func.return
}

// CHECK: physical.core @function()
%pe = physical.core @function() : () -> !physical.core