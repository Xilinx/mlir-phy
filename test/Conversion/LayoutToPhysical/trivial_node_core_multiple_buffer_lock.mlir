// RUN: phy-opt --convert-layout-to-physical="device=aie" %s | FileCheck %s

func.func private @kernel(%Q1: !spatial.queue<memref<1024xi32>>, %Q2: !spatial.queue<memref<1024xi32>>) {
  cf.br ^bb
^bb:

  %c0 = arith.constant 0 : index

  // CHECK: physical.lock_acquire<1> (%arg1)
  %0 = spatial.front(%Q1): memref<1024xi32>
  // CHECK: physical.lock_acquire<0> (%arg3)
  %1 = spatial.emplace(%Q2): memref<1024xi32>

  // CHECK: memref.load %arg0[%c0] : memref<1024xi32>
  %2 = memref.load %0[%c0]: memref<1024xi32>
  // CHECK: memref.load %arg2[%c0] : memref<1024xi32>
  %3 = memref.load %1[%c0]: memref<1024xi32>

  // CHECK: physical.lock_release<0> (%arg1)
  spatial.pop(%Q1: !spatial.queue<memref<1024xi32>>)
  // CHECK: physical.lock_release<1> (%arg3)
  spatial.push(%Q2: !spatial.queue<memref<1024xi32>>)
  cf.br ^bb
}

%Q1 = spatial.queue<1>(): !spatial.queue<memref<1024xi32>>
%Q2 = spatial.queue<1>(): !spatial.queue<memref<1024xi32>>
%node = spatial.node @kernel(%Q1, %Q2): (!spatial.queue<memref<1024xi32>>, !spatial.queue<memref<1024xi32>>) -> !spatial.node

layout.platform<"vck190"> {
  layout.device<"aie"> {
    
    // CHECK: physical.core @kernel1

    layout.place<"tile/6.3/id/1/buffer,tile/6.3/id/1/lock">(%Q1: !spatial.queue<memref<1024xi32>>)
    layout.place<"tile/6.3/id/2/buffer,tile/6.3/id/2/lock">(%Q2: !spatial.queue<memref<1024xi32>>)
    layout.place<"tile/6.3/core">(%node: !spatial.node)
    layout.route<[]>(%Q1: !spatial.queue<memref<1024xi32>> -> %node: !spatial.node)
    layout.route<[]>(%Q2: !spatial.queue<memref<1024xi32>> -> %node: !spatial.node)

  }
}
