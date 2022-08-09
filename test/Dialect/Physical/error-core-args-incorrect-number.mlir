// RUN: (phy-opt %s 2>&1 || true) | FileCheck %s

func.func @function() {
  func.return
}
%bus = physical.bus() : !physical.bus<i32>

// CHECK-LABEL: 'physical.core' op incorrect number of operands for callee
%pe = physical.core @function(%bus) : (!physical.bus<i32>) -> !physical.core