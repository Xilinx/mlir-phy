// RUN: phy-opt %s | FileCheck %s

// CHECK-LABEL: func.func @function
func.func @function(%lock: !physical.lock) {
  func.return
}

// CHECK: physical.lock
%lock = physical.lock<0>()

// CHECK: physical.core @function
%pe = physical.core @function(%lock)
    : (!physical.lock) -> !physical.core