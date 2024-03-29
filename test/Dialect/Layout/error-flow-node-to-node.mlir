// RUN: (phy-opt %s 2>&1 || true) | FileCheck %s
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

func.func @function(%q: !spatial.queue<memref<i32>>) {
  func.return
}
%queue = spatial.queue<2>(): !spatial.queue<memref<i32>>
%node1 = spatial.node @function(%queue)
    : (!spatial.queue<memref<i32>>) -> !spatial.node
%node2 = spatial.node @function(%queue)
    : (!spatial.queue<memref<i32>>) -> !spatial.node

layout.platform<"versal"> {
  layout.device<"aie"> {
    // CHECK: a node cannot be connected to a node using a flow
    layout.route<[]>(%node1: !spatial.node
                  -> %node2: !spatial.node)
  }
}