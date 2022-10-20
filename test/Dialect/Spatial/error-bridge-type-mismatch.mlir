// RUN: (phy-opt %s 2>&1 || true) | FileCheck %s
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

%queue1 = spatial.queue<2>(): !spatial.queue<memref<f32>>
%queue2 = spatial.queue<2>(): !spatial.queue<memref<i32>>

// CHECK: expects different type than prior uses
%bridge = spatial.bridge(%queue1 -> %queue2: !spatial.queue<memref<i32>>)
