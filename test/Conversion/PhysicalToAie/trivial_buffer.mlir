// REQUIRES: aie_found
// RUN: phy-opt --convert-physical-to-aie %s | FileCheck %s

// CHECK: %[[Tile:.*]] = AIE.tile(6, 3)

// CHECK: AIE.external_buffer 2147483647 : memref<1024xi32>
%0 = physical.buffer() { aie.external_address = "2203318222848" }: memref<1024xi32>

// CHECK: AIE.buffer(%[[Tile]]) : memref<1024xi32>
%1 = physical.buffer() { aie.tile = "6.3" }: memref<1024xi32>