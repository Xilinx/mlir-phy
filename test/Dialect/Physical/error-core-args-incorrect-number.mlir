// RUN: (phy-opt %s 2>&1 || true) | FileCheck %s
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

func.func @function() {
  func.return
}
%bus = physical.bus() : !physical.bus<i32>

// CHECK-LABEL: 'physical.core' op incorrect number of operands for callee
%pe = physical.core @function(%bus) : (!physical.bus<i32>) -> !physical.core