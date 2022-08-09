// RUN: phy-opt %s | FileCheck %s

// CHECK-LABEL: func.func @function(%arg0: !spatial.queue<memref<i32>>)
func.func @function(%q: !spatial.queue<memref<i32>>) {
  func.return
}

// CHECK: spatial.queue
%queue = spatial.queue<2>(): !spatial.queue<memref<i32>>

// CHECK: spatial.node @function
%node = spatial.node @function(%queue)
    : (!spatial.queue<memref<i32>>) -> !spatial.node

// CHECK: spatial.flow
%flow = spatial.flow(%queue: !spatial.queue<memref<i32>>
                  -> %node: !spatial.node)
      : !spatial.flow<memref<i32>>