// RUN: phy-opt %s | FileCheck %s

// CHECK: spatial.queue
%queue = spatial.queue<2>(): !spatial.queue<memref<i32>>

// CHECK: spatial.queue
%signal = spatial.queue<1>(): !spatial.queue<none>
