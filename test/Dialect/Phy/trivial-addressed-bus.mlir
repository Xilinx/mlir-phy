// RUN: phy-opt %s | FileCheck %s

// CHECK-LABEL: func.func @function
func.func @function(%bus: !phy.addressedBus<i32>) {
  %address = arith.constant 10 : index

  // CHECK: phy.startLoad
  %h0 = phy.startLoad %bus[%address] : !phy.addressedBus<i32>
  %v0 = phy.wait(%h0) : i32

  // CHECK: phy.startStore
  %h1 = phy.startStore %v0, %bus[%address] : !phy.addressedBus<i32>
  phy.wait(%h1) : none

  func.return
}

// CHECK: phy.buf
%buf = phy.buf() : memref<1024xi32>
// CHECK: phy.addressedBus
%abus = phy.addressedBus() : !phy.addressedBus<i32>
// CHECK: phy.mmap
phy.mmap(%abus[10:15], %buf[20:]: memref<1024xi32>)

// CHECK: phy.pe @function
%pe = phy.pe @function(%abus) : (!phy.addressedBus<i32>) -> !phy.pe
