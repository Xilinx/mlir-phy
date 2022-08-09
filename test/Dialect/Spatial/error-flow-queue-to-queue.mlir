// RUN: (phy-opt %s 2>&1 || true) | FileCheck %s

%queue1 = spatial.queue<2>(): !spatial.queue<memref<i32>>
%queue2 = spatial.queue<2>(): !spatial.queue<memref<i32>>

// CHECK: a queue cannot be connected to a queue using a flow
%flow = spatial.flow(%queue1: !spatial.queue<memref<i32>>
                  -> %queue2: !spatial.queue<memref<i32>>)
      : !spatial.flow<memref<i32>>