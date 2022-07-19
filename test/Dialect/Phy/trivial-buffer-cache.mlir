// RUN: phy-opt %s | FileCheck %s

// CHECK-LABEL: func.func @function(%arg0: memref<1024xi32>)
func.func @function(%buf: memref<1024xi32>) {
  func.return
}

// CHECK: phy.buf
%buf = phy.buf() : memref<1024xi32>
// CHECK: phy.pe @function
%pe = phy.pe @function(%buf) : (memref<1024xi32>) -> !phy.pe

// CHECK: phy.bus
%bus1 = phy.bus(%buf) : (memref<1024xi32>) -> !phy.bus<i32>
// CHECK: phy.bus
%bus2 = phy.bus(%pe) : (!phy.pe) -> !phy.bus<i32>
// CHECK: phy.cache
%cache = phy.cache(%bus1, %bus2) : !phy.cache<i32, 1024>
