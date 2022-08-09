// RUN: phy-opt %s | FileCheck %s

// CHECK: physical.bus
%bus1 = physical.bus(): !physical.bus<i32>
// CHECK: physical.bus
%bus2 = physical.bus(): !physical.bus<i32>
// CHECK: physical.cache
%cache = physical.cache(%bus1, %bus2) : !physical.cache<i32, 1024>
