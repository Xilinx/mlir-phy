// REQUIRES: aie_found
// RUN: phy-opt --convert-physical-to-aie %s | FileCheck %s

// CHECK: module {
// CHECK-NEXT: }
%0:2 = physical.stream<[0, 1]>(){ aie.tile = "6.0", aie.port = "DMA.O", aie.id = "0" }: (!physical.ostream<i32>, !physical.istream<i32>)