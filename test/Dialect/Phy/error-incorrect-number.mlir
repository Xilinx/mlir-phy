// RUN: (phy-opt %s 2>&1 || true) | FileCheck %s

func.func @function() {
  func.return
}
%1 = phy.pe @function()

// CHECK-LABEL: 'phy.pe' op incorrect number of operands for callee
%pe = phy.pe @function(%1)