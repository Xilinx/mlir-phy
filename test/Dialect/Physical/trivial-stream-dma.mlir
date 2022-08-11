// RUN: phy-opt %s | FileCheck %s

// CHECK: physical.buffer
%buffer1 = physical.buffer(): memref<1024xi32>

// CHECK: physical.lock
%lock1 = physical.lock<0>()

// CHECK: physical.buffer
%buffer2 = physical.buffer(): memref<1024xi32>

// CHECK: physical.lock
%lock2 = physical.lock<0>()

// CHECK: physical.stream
%stream = physical.stream(): !physical.stream<i32>

// CHECK: physical.istream
%input = physical.istream(%stream: !physical.stream<i32>)

// CHECK: physical.stream_dma
physical.stream_dma(%input: !physical.istream<i32>) {

  // CHECK: physical.stream_dma_connect
  ^bb0: physical.stream_dma_connect(
      %lock1[0->1], %buffer1[0:1024]: memref<1024xi32>, ^bb1)

  // CHECK: physical.stream_dma_connect
  ^bb1: physical.stream_dma_connect(
      %lock2[0->1], %buffer2[0:1024]: memref<1024xi32>, ^end)

  // CHECK: physical.end
  ^end: physical.end

}
