// RUN: phy-opt %s | FileCheck %s

// CHECK-LABEL: func.func @function()
func.func @function() {
  func.return
}

// CHECK: spatial.node @function()
%node = spatial.node @function() : () -> !spatial.node
