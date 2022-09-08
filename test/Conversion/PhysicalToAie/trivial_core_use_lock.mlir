// REQUIRES: aie_found
// RUN: phy-opt --convert-physical-to-aie %s | FileCheck %s

// CHECK: %[[Tile:.*]] = AIE.tile(6, 3)

// CHECK: %[[Lock:.*]] = AIE.lock(%[[Tile]], 0)
// CHECK: AIE.useLock(%[[Lock]], Release, 1)
%L = physical.lock<1>() { aie.tile = "6.3", aie.id = "0" }

func.func private @extern_kernel() -> ()

// CHECK: %[[BufB:.*]] = AIE.core(%[[Tile]]) {
// CHECK:   cf.br ^[[bb:.*]]
// CHECK: ^[[bb]]:
// CHECK:   AIE.useLock(%[[Lock]], Acquire, 1)
// CHECK:   func.call @extern_kernel
// CHECK:   AIE.useLock(%[[Lock]], Release, 0)
// CHECK:   AIE.useLock(%[[Lock]], Release, 1)
// CHECK:   cf.br ^[[bb:.*]]
// CHECK: }

func.func private @kernel(%0: !physical.lock) {
  physical.lock_acquire<1>(%0)
  func.call @extern_kernel() : () -> ()
  physical.lock_release<0>(%0)
  func.return
}

func.func private @kernel_wrapper(%0: !physical.lock) {
  cf.br ^br0
^br0:
  func.call @kernel(%0) : (!physical.lock) -> ()
  physical.lock_release<1>(%0)
  cf.br ^br0
}

physical.core @kernel_wrapper(%L) { aie.tile = "6.3" }: (!physical.lock) -> !physical.core
