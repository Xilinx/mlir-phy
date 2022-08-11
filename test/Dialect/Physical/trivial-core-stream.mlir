// RUN: phy-opt %s | FileCheck %s

// CHECK-LABEL: func.func @function
func.func @function(%in:  !physical.istream<i32>,
                    %out: !physical.ostream<i32>) {
  func.return
}

// CHECK: physical.stream
%stream = physical.stream(): !physical.stream<i32>

// CHECK: physical.istream
%input = physical.istream(%stream: !physical.stream<i32>)

// CHECK: physical.ostream
%output = physical.ostream(%stream: !physical.stream<i32>)

// CHECK: physical.core @function
%pe = physical.core @function(%input, %output)
    : (!physical.istream<i32>, !physical.ostream<i32>)
    -> !physical.core