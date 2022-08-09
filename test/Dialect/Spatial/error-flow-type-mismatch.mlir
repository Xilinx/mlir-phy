// RUN: (phy-opt %s 2>&1 || true) | FileCheck %s

func.func @function(%q: !spatial.queue<memref<i32>>) {
  func.return
}
%queue = spatial.queue<2>(): !spatial.queue<memref<i32>>
%node = spatial.node @function(%queue)
    : (!spatial.queue<memref<i32>>) -> !spatial.node

// CHECK: the datatype of the flow must match the queue
%flow = spatial.flow(%queue: !spatial.queue<memref<i32>>
                  -> %node: !spatial.node)
      : !spatial.flow<memref<f32>>