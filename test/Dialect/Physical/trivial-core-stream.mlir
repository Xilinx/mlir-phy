// RUN: phy-opt %s | FileCheck %s

// CHECK-LABEL: func.func @function
func.func @function(%in:  !physical.istream<i32>,
                    %out: !physical.ostream<i32>) {
  func.return
}

// CHECK: physical.stream
%stream:2 = physical.stream<[0, 1]>(): (!physical.ostream<i32>, !physical.istream<i32>)

// CHECK: physical.core @function
%pe = physical.core @function(%stream#1, %stream#0)
    : (!physical.istream<i32>, !physical.ostream<i32>)
    -> !physical.core