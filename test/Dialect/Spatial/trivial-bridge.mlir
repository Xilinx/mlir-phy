// RUN: phy-opt %s | FileCheck %s

// CHECK: spatial.queue
%queue1 = spatial.queue<2>(): !spatial.queue<memref<i32>>
// CHECK: spatial.queue
%queue2 = spatial.queue<2>(): !spatial.queue<memref<i32>>

// CHECK: spatial.bridge
%bridge = spatial.bridge(%queue1 -> %queue2: !spatial.queue<memref<i32>>)
