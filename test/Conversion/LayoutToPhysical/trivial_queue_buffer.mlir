// RUN: phy-opt --convert-layout-to-physical="device=aie" %s | FileCheck %s
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

%Q1 = spatial.queue<1>(): !spatial.queue<memref<1024xi32>>
%Q2 = spatial.queue<1>(): !spatial.queue<memref<1024xi32>>

layout.platform<"vck190"> {
  layout.device<"aie"> {
    
    // CHECK: physical.buffer() {aie.id = "1", aie.tile = "6.3"} : memref<1024xi32>
    layout.place<"tile/6.3/id/1/buffer">(%Q1: !spatial.queue<memref<1024xi32>>)

    // CHECK: physical.buffer() {aie.id = "2", aie.tile = "6.3"} : memref<1024xi32>
    layout.place<"tile/6.3/id/2/buffer">(%Q1: !spatial.queue<memref<1024xi32>>)
  }
}
