// RUN: phy-opt %s | FileCheck %s

// CHECK-LABEL: func.func @function()
func.func @function() {
  func.return
}

// CHECK: physical.core @function()
%pe = physical.core @function() : () -> !physical.core