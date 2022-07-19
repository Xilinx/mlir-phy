// RUN: phy-opt %s | FileCheck %s

// CHECK-LABEL: func.func @function(%arg0: memref<1024xi32>)
func.func @function(%mem: memref<1024xi32>) {
  %idx = arith.constant 0 : index

  // CHECK: memref.load
	%0 = memref.load %mem[%idx] : memref<1024xi32>
  // CHECK: memref.store
  memref.store %0, %mem[%idx] : memref<1024xi32>

  // CHECK: phy.startLoad
  %h0 = phy.startLoad %mem[%idx] : memref<1024xi32>
  // CHECK: phy.startStore
  %h1 = phy.startStore %0, %mem[%idx] : memref<1024xi32>
  // CHECK: phy.wait
	%1 = phy.wait(%h0 : !phy.handle<i32>)
  // CHECK: phy.wait
	phy.wait(%h1 : !phy.handle<none>)
  func.return
}

// CHECK: phy.buf
%buf = phy.buf() : memref<1024xi32>
// CHECK: phy.pe @function
%pe1 = phy.pe @function(%buf) : (memref<1024xi32>) -> !phy.pe
// CHECK: phy.pe @function
%pe2 = phy.pe @function(%buf) : (memref<1024xi32>) -> !phy.pe