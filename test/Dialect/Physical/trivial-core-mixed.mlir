// RUN: phy-opt %s | FileCheck %s

// CHECK-LABEL: func.func @function(%arg0: memref<1024xi32>, %arg1: !physical.bus<i32>)
func.func @function(%buf: memref<1024xi32>, %net: !physical.bus<i32>) {
  func.return
}

// CHECK: physical.buffer
%buf = physical.buffer() : memref<1024xi32>
// CHECK: physical.bus
%bus = physical.bus() : !physical.bus<i32>
// CHECK: physical.core @function
%pe1 = physical.core @function(%buf, %bus) : (memref<1024xi32>, !physical.bus<i32>) -> !physical.core