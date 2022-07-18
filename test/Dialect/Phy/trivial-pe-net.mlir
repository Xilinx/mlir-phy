// RUN: phy-opt %s | FileCheck %s

// CHECK-LABEL: func.func @function(%arg0: !phy.net<i32>)
func.func @function(%net: !phy.net<i32>) {
  func.return
}

// CHECK: phy.net
%net = phy.net() : !phy.net<i32>
// CHECK: phy.pe @function
%pe1 = phy.pe @function(%net) : (!phy.net<i32>) -> !phy.pe
// CHECK: phy.pe @function
%pe2 = phy.pe @function(%net) : (!phy.net<i32>) -> !phy.pe