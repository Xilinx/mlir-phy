// RUN: phy-opt %s | FileCheck %s

// CHECK-LABEL: func.func @func
func.func @func(%buf: memref<1024xi32>, %net: !phy.net<i32>) {
  func.return
}
// CHECK: phy.buf
%buf = phy.buf() : memref<1024xi32>
// CHECK: phy.net
%net1   = phy.net() : !phy.net<i32>
// CHECK: phy.net
%net2   = phy.net() : !phy.net<i32>
// CHECK: phy.router
%router = phy.router(%net1, %net2) : !phy.router<i32, 2>
// CHECK: phy.pe
%pe     = phy.pe @func(%buf, %net1) : (memref<1024xi32>, !phy.net<i32>) -> !phy.pe

// CHECK: phy.bus
%bus1   = phy.bus(%buf) : (memref<1024xi32>) -> !phy.bus<i32>
// CHECK: phy.bus
%bus2   = phy.bus(%pe) : (!phy.pe) -> !phy.bus<i32>
// CHECK: phy.cache
%cache  = phy.cache(%bus1, %bus2) : !phy.cache<i32, 1024>

// CHECK: phy.platform<"xilinx">
phy.platform<"xilinx"> {
  // CHECK: phy.device<"hls">
  phy.device<"hls"> {
    // CHECK: phy.place<"slr0">
    phy.place<"slr0">(%buf : memref<1024xi32>)
    // CHECK: phy.route<["slr0-slr1"]>
		phy.route<["slr0-slr1"]>(%bus1 : !phy.bus<i32>)
    // CHECK: phy.place<"slr1">
    phy.place<"slr1">(%cache : !phy.cache<i32, 1024>)
    // CHECK: phy.route<[]>
		phy.route<[]>(%bus2 : !phy.bus<i32>)  // local memory access
    // CHECK: phy.place<"slr1">
		phy.place<"slr1">(%pe : !phy.pe) 
  }
  // CHECK: phy.device<"aie">
  phy.device<"aie"> {
    // CHECK: phy.place<"shimswitchbox">
    phy.place<"shimswitchbox">(%router : !phy.router<i32, 2>)
    // CHECK: phy.route<["tile70-tile71", "tile71-tile72"]>
		phy.route<["tile70-tile71", "tile71-tile72"]>(%net2 : !phy.net<i32>)
  }
  // CHECK: phy.route<["pl-aie"]>
	phy.route<["pl-aie"]>(%net1 : !phy.net<i32>)
}

// CHECK: phy.net
%net = phy.net() : !phy.net<i32>
// CHECK: phy.route<["pcie"]>
phy.route<["pcie"]>(%net : !phy.net<i32>)
