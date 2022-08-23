// RUN: phy-opt --convert-physical-to-aie %s | FileCheck %s

// CHECK: %[[Tile:.*]] = AIE.tile(6, 0)

// CHECK: %[[Lock:.*]] = AIE.lock(%[[Tile]], 0)
// CHECK: AIE.useLock(%[[Lock]], Release, 1)
%0 = physical.lock<1>() { aie.tile = "6.0", aie.id = "0" }
