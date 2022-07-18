// RUN: (phy-opt %s 2>&1 || true) | FileCheck %s

func.func @function(%buf: memref<1024xi32>) {
  func.return
}
%buf = phy.buf() : memref<1024xi32>
%pe = phy.pe @function(%buf) : (memref<1024xi32>) -> !phy.pe

// CHECK-LABEL: 'phy.bus' op endpoints must have the same base type as the bus
%bus = phy.bus(%buf, %pe) : (memref<1024xi32>, !phy.pe) -> !phy.bus<f32>