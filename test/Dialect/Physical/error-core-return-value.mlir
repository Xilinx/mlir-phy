// RUN: (phy-opt %s 2>&1 || true) | FileCheck %s

func.func @function() -> i1 {
  %0 = llvm.mlir.constant(0) : i1
  func.return %0 : i1
}

// CHECK-LABEL: 'physical.core' op callee cannot have a return value
%pe = physical.core @function() : () -> !physical.core