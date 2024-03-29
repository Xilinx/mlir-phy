// RUN: phy-opt --convert-layout-to-physical="device=aie" %s | FileCheck %s
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

func.func private @kernel(%Q: !spatial.queue<memref<1024xi32>>) {
  cf.br ^bb
^bb:
  cf.br ^bb
}

%Q = spatial.queue<1>(): !spatial.queue<memref<1024xi32>>
%node = spatial.node @kernel(%Q): (!spatial.queue<memref<1024xi32>>) -> !spatial.node

layout.platform<"vck190"> {
  layout.device<"aie"> {
 
    // CHECK: %[[Buffer:.*]] = physical.buffer() {aie.id = "1", aie.tile = "6.3"} : memref<1024xi32>
    // CHECK: %[[Lock:.*]] = physical.lock<0> () {aie.id = "2", aie.tile = "6.3"}
    // CHECK: physical.core @kernel1(%[[Buffer]], %[[Lock]]) {aie.tile = "6.3"} : (memref<1024xi32>, !physical.lock) -> !physical.core

    layout.place<"tile/6.2/id/1/buffer,tile/6.2/id/2/lock">(%Q: !spatial.queue<memref<1024xi32>>)
    layout.place<"tile/6.3/core">(%node: !spatial.node)
    layout.route<["tile/6.3/id/1/buffer,tile/6.3/id/2/lock"]>(%Q: !spatial.queue<memref<1024xi32>> -> %node: !spatial.node)

  }
}
