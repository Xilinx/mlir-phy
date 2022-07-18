// RUN: phy-opt %s | FileCheck %s

// CHECK-LABEL: func.func @function(%arg0: memref<1024xi32>)
func.func @function(%buf: memref<1024xi32>) {
  func.return
}

// CHECK: phy.buf
%buf = phy.buf() : memref<1024xi32>
// CHECK: phy.pe @function
%pe1 = phy.pe @function(%buf) : (memref<1024xi32>) -> !phy.pe
// CHECK: phy.pe @function
%pe2 = phy.pe @function(%buf) : (memref<1024xi32>) -> !phy.pe