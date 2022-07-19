// RUN: (phy-opt %s 2>&1 || true) | FileCheck %s

func.func @function(%buf: memref<1024xi32>) {
  func.return
}
%buf = phy.buf() : memref<1024xi32>
%pe = phy.pe @function(%buf) : (memref<1024xi32>) -> !phy.pe
%bus1 = phy.bus(%buf) : (memref<1024xi32>) -> !phy.bus<i32>
%bus2 = phy.bus(%pe) : (!phy.pe) -> !phy.bus<i32>

// CHECK: expects different type than prior uses: '!phy.bus<i16>' vs '!phy.bus<i32>'
%cache = phy.cache(%bus1, %bus2) : !phy.cache<i16, 1024>
