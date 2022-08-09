// RUN: phy-opt %s | FileCheck %s

// CHECK-LABEL: func.func @function(%arg0: memref<1024xi32>)
func.func @function(%buf: memref<1024xi32>) {
  func.return
}

// CHECK: physical.buffer
%buf = physical.buffer() : memref<1024xi32>
// CHECK: physical.core @function
%pe = physical.core @function(%buf) : (memref<1024xi32>) -> !physical.core