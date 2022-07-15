// RUN: phy-opt %s | FileCheck %s

// CHECK-LABEL: func.func @function() -> i1
func.func @function() -> i1 {
  %0 = llvm.mlir.constant(false) : i1
  func.return %0 : i1
}