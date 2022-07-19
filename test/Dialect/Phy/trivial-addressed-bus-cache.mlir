// RUN: phy-opt %s | FileCheck %s

// CHECK-LABEL: phy.addressedBus
%abus1 = phy.addressedBus() : !phy.addressedBus<i32>
// CHECK: phy.addressedBus
%abus2 = phy.addressedBus() : !phy.addressedBus<i32>
// CHECK: phy.addressedCache
%cache  = phy.addressedCache(%abus1, %abus2) : !phy.cache<i32, 1024>
