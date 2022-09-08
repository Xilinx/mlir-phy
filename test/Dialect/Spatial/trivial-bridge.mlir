// RUN: phy-opt %s | FileCheck %s
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// CHECK: spatial.queue
%queue1 = spatial.queue<2>(): !spatial.queue<memref<i32>>
// CHECK: spatial.queue
%queue2 = spatial.queue<2>(): !spatial.queue<memref<i32>>

// CHECK: spatial.bridge
%bridge = spatial.bridge(%queue1 -> %queue2: !spatial.queue<memref<i32>>)
