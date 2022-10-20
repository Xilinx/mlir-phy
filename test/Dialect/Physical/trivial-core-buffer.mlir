// RUN: phy-opt %s | FileCheck %s
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @function(%arg0: memref<1024xi32>)
func.func @function(%buf: memref<1024xi32>) {
  func.return
}

// CHECK: physical.buffer
%buf = physical.buffer() : memref<1024xi32>
// CHECK: physical.core @function
%pe = physical.core @function(%buf) : (memref<1024xi32>) -> !physical.core