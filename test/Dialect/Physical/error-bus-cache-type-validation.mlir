// RUN: (phy-opt %s 2>&1 || true) | FileCheck %s

%bus1 = physical.bus(): !physical.bus<i32>
%bus2 = physical.bus(): !physical.bus<i32>

// CHECK: expects different type than prior uses: '!physical.bus<i16>' vs '!physical.bus<i32>'
%cache = physical.bus_cache(%bus1, %bus2) : !physical.bus_cache<i16, 1024>