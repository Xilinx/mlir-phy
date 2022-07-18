// RUN: phy-opt %s | FileCheck %s

// CHECK-LABEL: func.func @function(%arg0: !phy.net<i32>)
func.func @function(%net: !phy.net<i32>) {
  // CHECK: phy.valid
  %b1 = phy.valid(%net : !phy.net<i32>, 0)
  // CHECK: phy.ready
  %b2 = phy.ready(%net : !phy.net<i32>, 0)
  // CHECK: phy.pop
  %v3 = phy.pop(%net : !phy.net<i32>, 0)
  // CHECK: phy.push
  phy.push(%v3 : i32, %net : !phy.net<i32>, 0)
  func.return
}

// CHECK: phy.net
%net = phy.net() : !phy.net<i32>
// CHECK: phy.pe @function
%pe1 = phy.pe @function(%net) : (!phy.net<i32>) -> !phy.pe