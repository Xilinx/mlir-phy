// RUN: phy-opt %s | FileCheck %s

// CHECK: physical.bus
%bus1 = physical.bus(): !physical.bus<i32>
// CHECK: physical.bus
%bus2 = physical.bus(): !physical.bus<i32>
// CHECK: physical.bus_cache
%cache = physical.bus_cache(%bus1, %bus2) : !physical.bus_cache<i32, 1024>
