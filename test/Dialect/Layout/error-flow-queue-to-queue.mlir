// RUN: (phy-opt %s 2>&1 || true) | FileCheck %s
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

%queue1 = spatial.queue<2>(): !spatial.queue<memref<i32>>
%queue2 = spatial.queue<2>(): !spatial.queue<memref<i32>>

layout.platform<"versal"> {
  layout.device<"aie"> {
    // CHECK: a queue cannot be connected to a queue using a flow
    layout.route<[]>(%queue1: !spatial.queue<memref<i32>>
                  -> %queue2: !spatial.queue<memref<i32>>)
  }
}