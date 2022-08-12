// RUN: phy-opt %s | FileCheck %s

// CHECK-LABEL: func.func @function
func.func @function(%buf: memref<1024xi32>,
                    %net: !physical.bus<i32>,
                    %in:  !physical.istream<i32>,
                    %out: !physical.ostream<i32>) {
  func.return
}

// CHECK: physical.stream
%stream:2 = physical.stream<[0, 1]>(): (!physical.ostream<i32>, !physical.istream<i32>)

// CHECK: physical.buffer
%buf = physical.buffer() : memref<1024xi32>

// CHECK: physical.bus
%bus = physical.bus() : !physical.bus<i32>

// CHECK: physical.core @function
%pe1 = physical.core @function(%buf, %bus, %stream#1, %stream#0)
     : (memref<1024xi32>, !physical.bus<i32>,
        !physical.istream<i32>, !physical.ostream<i32>)
     -> !physical.core