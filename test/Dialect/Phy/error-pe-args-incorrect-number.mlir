// RUN: (phy-opt %s 2>&1 || true) | FileCheck %s

func.func @function() {
  func.return
}
%net = phy.net() : !phy.net<i32>

// CHECK-LABEL: 'phy.pe' op incorrect number of operands for callee
%pe = phy.pe @function(%net) : (!phy.net<i32>) -> !phy.pe