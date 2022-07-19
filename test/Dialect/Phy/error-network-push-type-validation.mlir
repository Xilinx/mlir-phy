// RUN: (phy-opt %s 2>&1 || true) | FileCheck %s

func.func @function(%net: !phy.net<i32>) {
  %0 = llvm.mlir.constant(0) : i1
  // CHECK-LABEL: expects different type than prior uses: '!phy.net<i1>' vs '!phy.net<i32>'
  phy.push(%0 : i1, %net, 0)
  func.return
}

%net = phy.net() : !phy.net<i32>
%pe1 = phy.pe @function(%net) : (!phy.net<i32>) -> !phy.pe