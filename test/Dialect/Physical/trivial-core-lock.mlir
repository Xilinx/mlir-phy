// RUN: phy-opt %s | FileCheck %s
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @function
func.func @function(%lock: !physical.lock) {
  func.return
}

// CHECK: physical.lock
%lock = physical.lock<0>()

// CHECK: physical.core @function
%pe = physical.core @function(%lock)
    : (!physical.lock) -> !physical.core