// RUN: phy-opt %s | FileCheck %s
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @function
func.func @function(%in:  !physical.istream<i32>,
                    %out: !physical.ostream<i32>) {
  func.return
}

// CHECK: physical.stream
%stream:2 = physical.stream<[0, 1]>(): (!physical.ostream<i32>, !physical.istream<i32>)

// CHECK: physical.core @function
%pe = physical.core @function(%stream#1, %stream#0)
    : (!physical.istream<i32>, !physical.ostream<i32>)
    -> !physical.core