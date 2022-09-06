// RUN: phy-opt --convert-layout-to-physical="device=aie" %s | FileCheck %s

func.func private @kernel(%Q: !spatial.queue<memref<1024xi32>>) {
  cf.br ^bb
^bb:
  %0 = spatial.front(%Q): memref<1024xi32>
  spatial.pop(%Q: !spatial.queue<memref<1024xi32>>)
  cf.br ^bb
}

%Q = spatial.queue<1>(): !spatial.queue<memref<1024xi32>>
%node = spatial.node @kernel(%Q): (!spatial.queue<memref<1024xi32>>) -> !spatial.node

layout.platform<"vck190"> {
  layout.device<"aie"> {
    
    // CHECK: physical.core @kernel1(%0, %1) {aie.tile = "6.3"} : (memref<1024xi32>, !physical.lock) -> !physical.core

    layout.place<"tile/6.3/id/1/buffer,tile/6.3/id/2/lock">(%Q: !spatial.queue<memref<1024xi32>>)
    layout.place<"tile/6.3/core">(%node: !spatial.node)
    layout.route<[]>(%Q: !spatial.queue<memref<1024xi32>> -> %node: !spatial.node)

  }
}
