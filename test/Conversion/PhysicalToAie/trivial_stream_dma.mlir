// RUN: phy-opt --convert-physical-to-aie %s | FileCheck %s

// CHECK: %[[Tile:.*]] = AIE.tile(6, 0)

// CHECK: %[[Lock:.*]] = AIE.lock(%[[Tile]], 0)

%0:2 = physical.stream<[0, 1]>(){ aie.tile = "6.0", aie.port = "DMA.O", aie.id = 0 }: (!physical.ostream<i32>, !physical.istream<i32>)
%1   = physical.buffer() { aie.external_address = "2203318222848" }: memref<1024xi32>
%2   = physical.lock<0>() { aie.tile = "6.0", aie.id = "0" }

// CHECK: AIE.shimDMA(%[[Tile]])

physical.stream_dma(%0#0: !physical.ostream<i32>) {
  // TODO: %m = physical.stream_dma_connect<0>(%2[1->0], %1[0:1024]: memref<1024xi32>, %m)
} { aie.tile = "6.0", aie.engine = "MM2S", aie.id = "0" }