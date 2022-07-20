// RUN: phy-opt %s | FileCheck %s

// CHECK-LABEL: func.func @function(%arg0: !phy.net<i32>)
func.func @function(%net: !phy.net<i32>) {
  // CHECK: phy.valid
  %b1 = phy.valid<0>(%net : !phy.net<i32>)
  // CHECK: phy.ready
  %b2 = phy.ready<0>(%net : !phy.net<i32>)
  // CHECK: phy.pop<0>
  %v3 = phy.pop<0>(%net) : i32
  // CHECK: phy.push<0>
  phy.push<0>(%v3 : i32, %net)
  func.return
}

// CHECK: phy.net
%net = phy.net() : !phy.net<i32>
// CHECK: phy.pe @function
%pe1 = phy.pe @function(%net) : (!phy.net<i32>) -> !phy.pe