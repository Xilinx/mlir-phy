// REQUIRES: aie_found
// RUN: phy-opt --convert-physical-to-aie %s | FileCheck %s

// CHECK: %[[Tile:.*]] = AIE.tile(6, 3)

// CHECK: %[[BufA:.*]] = AIE.buffer(%[[Tile]])
// CHECK: %[[BufB:.*]] = AIE.buffer(%[[Tile]])
%A  = physical.buffer() { aie.tile = "6.3" }: memref<1024xi32>
%B  = physical.buffer() { aie.tile = "6.3" }: memref<1024xi32>

// CHECK: %[[Lock:.*]] = AIE.lock(%[[Tile]], 0)
// CHECK: AIE.useLock(%[[Lock]], Release, 1)
%L = physical.lock<1>() { aie.tile = "6.3", aie.id = "0" }

func.func private @extern_kernel(%0: memref<1024xi32>, %1: memref<1024xi32>) -> ()

func.func private @kernel(%0: memref<1024xi32>, %1: memref<1024xi32>) {
  cf.br ^bb
^bb:
  func.call @extern_kernel(%0, %1) : (memref<1024xi32>, memref<1024xi32>) -> ()
  cf.br ^bb
}

physical.core @kernel(%A, %B) { aie.tile = "6.3" }: (memref<1024xi32>, memref<1024xi32>) -> !physical.core
