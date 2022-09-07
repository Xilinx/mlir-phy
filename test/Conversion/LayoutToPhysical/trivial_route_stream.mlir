// RUN: phy-opt --convert-layout-to-physical="device=aie" %s | FileCheck %s

func.func private @kernel(%Q: !spatial.queue<memref<1024xi32>>) {
  cf.br ^bb
^bb:
  cf.br ^bb
}

%Q = spatial.queue<1>(): !spatial.queue<memref<1024xi32>>
%node = spatial.node @kernel(%Q): (!spatial.queue<memref<1024xi32>>) -> !spatial.node

layout.platform<"vck190"> {
  layout.device<"aie"> {
 
    // CHECK: physical.stream () {aie.id = "0", aie.port = "DMA.O", aie.tile = "6.2"} : (!physical.ostream<i32>, !physical.istream<i32>

    layout.route<["tile/6.2/port/DMA.O/id/0/stream"]>
                  (%Q: !spatial.queue<memref<1024xi32>> -> %node: !spatial.node)

  }
}
