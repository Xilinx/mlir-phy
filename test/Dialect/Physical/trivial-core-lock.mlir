// RUN: phy-opt %s | FileCheck %s
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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