// RUN: phy-opt %s | FileCheck %s

// CHECK-LABEL: func.func @function(%arg0: memref<1024xi32>, %arg1: !phy.net<i32>)
func.func @function(%buf: memref<1024xi32>, %net: !phy.net<i32>) {
  func.return
}

// CHECK: phy.buf
%buf = phy.buf() : memref<1024xi32>
// CHECK: phy.net
%net = phy.net() : !phy.net<i32>
// CHECK: phy.pe @function
%pe1 = phy.pe @function(%buf, %net) : (memref<1024xi32>, !phy.net<i32>) -> !phy.pe