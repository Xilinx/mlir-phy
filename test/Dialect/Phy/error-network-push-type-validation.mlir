// RUN: (phy-opt %s 2>&1 || true) | FileCheck %s

func.func @function(%net: !phy.net<i32>) {
  %0 = llvm.mlir.constant(0) : i1
  // CHECK-LABEL: 'phy.push' op data must have the same type as the network
  phy.push(%0 : i1, %net : !phy.net<i32>, 0)
  func.return
}

%net = phy.net() : !phy.net<i32>
%pe1 = phy.pe @function(%net) : (!phy.net<i32>) -> !phy.pe