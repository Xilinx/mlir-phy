// RUN: phy-opt %s | FileCheck %s

// CHECK-LABEL: func.func @function(%arg0: !phy.net<i32>)
func.func @function(%net: !phy.net<i32>) {
  func.return
}

// CHECK: phy.net
%net1 = phy.net() : !phy.net<i32>
// CHECK: phy.pe @function
%pe1 = phy.pe @function(%net1) : (!phy.net<i32>) -> !phy.pe

// CHECK: phy.net
%net2 = phy.net() : !phy.net<i32>
// CHECK: phy.pe @function
%pe2 = phy.pe @function(%net2) : (!phy.net<i32>) -> !phy.pe

// CHECK: phy.router
%router = phy.router(%net1, %net2) : !phy.router<i32, 2>
