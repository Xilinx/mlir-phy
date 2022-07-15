// RUN: phy-opt %s | FileCheck %s

// CHECK-LABEL: func.func @function()
func.func @function() {
  func.return
}

// CHECK: phy.pe @function()
%pe = phy.pe @function() : () -> !phy.pe