// RUN: (phy-opt %s 2>&1 || true) | FileCheck %s

func.func @function(%data : i32) {
  func.return
}
%1 = llvm.mlir.constant(1) : i32

// CHECK-LABEL: 'phy.pe' op operand {{.*}} must be {{.*}}, but got 'i32'
%pe = phy.pe @function(%1) : (i32) -> !phy.pe