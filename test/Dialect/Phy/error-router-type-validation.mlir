// RUN: (phy-opt %s 2>&1 || true) | FileCheck %s

func.func @function(%net: !phy.net<i32>) {
  func.return
}
%net1 = phy.net() : !phy.net<i32>
%pe1 = phy.pe @function(%net1) : (!phy.net<i32>) -> !phy.pe
%net2 = phy.net() : !phy.net<i32>
%pe2 = phy.pe @function(%net2) : (!phy.net<i32>) -> !phy.pe

// CHECK: expects different type than prior uses: '!phy.net<i16>' vs '!phy.net<i32>'
%router = phy.router(%net1, %net2) : !phy.router<i16, 2>
