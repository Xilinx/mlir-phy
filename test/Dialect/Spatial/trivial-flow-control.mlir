// RUN: phy-opt %s | FileCheck %s

// CHECK-LABEL: func.func @function(%arg0: !spatial.queue<memref<i32>>)
func.func @function(%q: !spatial.queue<memref<i32>>) {
  // CHECK: spatial.valid
  %00 = spatial.valid(%q : !spatial.queue<memref<i32>>)
  // CHECK: spatial.valid
  %01 = spatial.valid<1>(%q : !spatial.queue<memref<i32>>)
  // CHECK: spatial.full
  %10 = spatial.full(%q : !spatial.queue<memref<i32>>)
  // CHECK: spatial.full
  %11 = spatial.full<1>(%q : !spatial.queue<memref<i32>>)
  // CHECK: spatial.front
  %accessor0 = spatial.front(%q) : memref<i32>
  // CHECK: spatial.front
  %accessor1 = spatial.front<1>(%q) : memref<i32>
  // CHECK: spatial.emplace
  %writer0 = spatial.emplace(%q) : memref<i32>
  // CHECK: spatial.emplace
  %writer1 = spatial.emplace<1>(%q) : memref<i32>
  // CHECK: spatial.pop
  spatial.pop(%q: !spatial.queue<memref<i32>>)
  // CHECK: spatial.pop
  spatial.pop<1>(%q: !spatial.queue<memref<i32>>)
  // CHECK: spatial.push
  spatial.push(%q: !spatial.queue<memref<i32>>)
  // CHECK: spatial.push
  spatial.push<1>(%q: !spatial.queue<memref<i32>>)
  return
}

// CHECK: spatial.queue
%queue = spatial.queue<2>(): !spatial.queue<memref<i32>>

// CHECK: spatial.node @function
%node = spatial.node @function(%queue)
    : (!spatial.queue<memref<i32>>) -> !spatial.node