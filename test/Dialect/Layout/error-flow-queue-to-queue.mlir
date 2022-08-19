// RUN: (phy-opt %s 2>&1 || true) | FileCheck %s

%queue1 = spatial.queue<2>(): !spatial.queue<memref<i32>>
%queue2 = spatial.queue<2>(): !spatial.queue<memref<i32>>

layout.platform<"versal"> {
  layout.device<"aie"> {
    // CHECK: a queue cannot be connected to a queue using a flow
    layout.route<[]>(%queue1: !spatial.queue<memref<i32>>
                  -> %queue2: !spatial.queue<memref<i32>>)
  }
}