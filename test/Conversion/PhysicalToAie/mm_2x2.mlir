// RUN: phy-opt %s | FileCheck %s

// CHECK-LABEL: module @MM_2x2
module @MM_2x2 {

  %eA_0   = physical.buffer() { aie.external_address = 0x020100000000 }: memref<1024xi32>
  %eA_1   = physical.buffer() { aie.external_address = 0x020100001000 }: memref<1024xi32>
  %eB_0_0 = physical.buffer() { aie.external_address = 0x020100002000 }: memref<1024xi32>
  %eB_0_1 = physical.buffer() { aie.external_address = 0x020100003000 }: memref<1024xi32>
  %eB_1_0 = physical.buffer() { aie.external_address = 0x020100004000 }: memref<1024xi32>
  %eB_1_1 = physical.buffer() { aie.external_address = 0x020100005000 }: memref<1024xi32>
  %eC_0   = physical.buffer() { aie.external_address = 0x020100006000 }: memref<1024xi32>
  %eC_1   = physical.buffer() { aie.external_address = 0x020100007000 }: memref<1024xi32>
  %A_0_a  = physical.buffer() { aie.tile = [6, 3] }: memref<1024xi32>
  %A_0_b  = physical.buffer() { aie.tile = [7, 3] }: memref<1024xi32>
  %A_1_a  = physical.buffer() { aie.tile = [6, 4] }: memref<1024xi32>
  %A_1_b  = physical.buffer() { aie.tile = [7, 4] }: memref<1024xi32>
  %B_0_0  = physical.buffer() { aie.tile = [6, 3] }: memref<1024xi32>
  %B_0_1  = physical.buffer() { aie.tile = [6, 4] }: memref<1024xi32>
  %B_1_0  = physical.buffer() { aie.tile = [7, 3] }: memref<1024xi32>
  %B_1_1  = physical.buffer() { aie.tile = [7, 4] }: memref<1024xi32>
  %S_0_0  = physical.buffer() { aie.tile = [6, 3] }: memref<1024xi32>
  %S_0_1  = physical.buffer() { aie.tile = [6, 3] }: memref<1024xi32>
  %S_1_0  = physical.buffer() { aie.tile = [7, 3] }: memref<1024xi32>
  %S_1_1  = physical.buffer() { aie.tile = [7, 3] }: memref<1024xi32>
  %C_0    = physical.buffer() { aie.tile = [6, 4] }: memref<1024xi32>
  %C_1    = physical.buffer() { aie.tile = [7, 4] }: memref<1024xi32>

  %leA_0   = physical.lock<0>() { aie.tile = [6, 0], aie.id = 0 }
  %leA_1   = physical.lock<0>() { aie.tile = [6, 0], aie.id = 1 }
  %leB_0_0 = physical.lock<0>() { aie.tile = [6, 0], aie.id = 2 }
  %leB_0_1 = physical.lock<0>() { aie.tile = [6, 0], aie.id = 3 }
  %leB_1_0 = physical.lock<0>() { aie.tile = [7, 0], aie.id = 0 }
  %leB_1_1 = physical.lock<0>() { aie.tile = [7, 0], aie.id = 1 }
  %leC_0   = physical.lock<0>() { aie.tile = [7, 0], aie.id = 2 }
  %leC_1   = physical.lock<0>() { aie.tile = [7, 0], aie.id = 3 }
  %lA_0_a  = physical.lock<0>() { aie.tile = [6, 3], aie.id = 0 }
  %lA_0_b  = physical.lock<0>() { aie.tile = [7, 3], aie.id = 0 }
  %lA_1_a  = physical.lock<0>() { aie.tile = [6, 4], aie.id = 0 }
  %lA_1_b  = physical.lock<0>() { aie.tile = [7, 4], aie.id = 0 }
  %lB_0_0  = physical.lock<0>() { aie.tile = [6, 3], aie.id = 1 }
  %lB_0_1  = physical.lock<0>() { aie.tile = [6, 4], aie.id = 1 }
  %lB_1_0  = physical.lock<0>() { aie.tile = [7, 3], aie.id = 1 }
  %lB_1_1  = physical.lock<0>() { aie.tile = [7, 4], aie.id = 1 }
  %lS_0_0  = physical.lock<1>() { aie.tile = [6, 3], aie.id = 2 }
  %lS_0_1  = physical.lock<0>() { aie.tile = [6, 3], aie.id = 3 }
  %lS_1_0  = physical.lock<1>() { aie.tile = [7, 3], aie.id = 2 }
  %lS_1_1  = physical.lock<0>() { aie.tile = [7, 3], aie.id = 3 }
  %lC_0    = physical.lock<0>() { aie.tile = [6, 4], aie.id = 2 }
  %lC_1    = physical.lock<0>() { aie.tile = [7, 4], aie.id = 2 }

  %sA:2      = physical.stream<[0, 1]>(){ aie.tile = [6, 0], aie.port = "O.DMA.0" }: (!physical.ostream<i32>, !physical.istream<i32>)
  %sA_0_a:2  = physical.stream<[0]>()   { aie.tile = [6, 3], aie.port = "I.DMA.0" }: (!physical.ostream<i32>, !physical.istream<i32>)
  %sA_0_b:2  = physical.stream<[0]>()   { aie.tile = [7, 3], aie.port = "I.DMA.0" }: (!physical.ostream<i32>, !physical.istream<i32>)
  %sA_1_a:2  = physical.stream<[1]>()   { aie.tile = [6, 4], aie.port = "I.DMA.0" }: (!physical.ostream<i32>, !physical.istream<i32>)
  %sA_1_b:2  = physical.stream<[1]>()   { aie.tile = [7, 4], aie.port = "I.DMA.0" }: (!physical.ostream<i32>, !physical.istream<i32>)
  %sB_0:2    = physical.stream<[2, 3]>(){ aie.tile = [6, 0], aie.port = "O.DMA.1" }: (!physical.ostream<i32>, !physical.istream<i32>)
  %sB_0_0:2  = physical.stream<[2]>()   { aie.tile = [6, 3], aie.port = "I.DMA.1" }: (!physical.ostream<i32>, !physical.istream<i32>)
  %sB_0_1:2  = physical.stream<[3]>()   { aie.tile = [6, 4], aie.port = "I.DMA.1" }: (!physical.ostream<i32>, !physical.istream<i32>)
  %sB_1:2    = physical.stream<[4, 5]>(){ aie.tile = [7, 0], aie.port = "O.DMA.0" }: (!physical.ostream<i32>, !physical.istream<i32>)
  %sB_1_0:2  = physical.stream<[4]>()   { aie.tile = [7, 3], aie.port = "I.DMA.1" }: (!physical.ostream<i32>, !physical.istream<i32>)
  %sB_1_1:2  = physical.stream<[5]>()   { aie.tile = [7, 4], aie.port = "I.DMA.1" }: (!physical.ostream<i32>, !physical.istream<i32>)
  %sC_0:2    = physical.stream<[6]>()   { aie.tile = [6, 4], aie.port = "O.DMA.0" }: (!physical.ostream<i32>, !physical.istream<i32>)
  %seC_0:2   = physical.stream<[6]>()   { aie.tile = [7, 0], aie.port = "I.DMA.0" }: (!physical.ostream<i32>, !physical.istream<i32>)
  %sC_1:2    = physical.stream<[7]>()   { aie.tile = [7, 4], aie.port = "O.DMA.0" }: (!physical.ostream<i32>, !physical.istream<i32>)
  %seC_1:2   = physical.stream<[7]>()   { aie.tile = [7, 0], aie.port = "I.DMA.1" }: (!physical.ostream<i32>, !physical.istream<i32>)

  physical.stream_dma(%sA#0: !physical.ostream<i32>) {
    %0 = physical.stream_dma_connect<0>(%leA_0[1->0], %eA_0[0:1024]: memref<1024xi32>, %1)
    %1 = physical.stream_dma_connect<1>(%leA_1[1->0], %eA_1[0:1024]: memref<1024xi32>, %0)
  } { aie.tile = [6, 0], aie.engine = "MM2S0" }
  physical.stream_dma(%sA_0_a#1: !physical.istream<i32>) {
    %0 = physical.stream_dma_connect(%lA_0_a[0->1], %A_0_a[0:1024]: memref<1024xi32>, %0)
  } { aie.tile = [6, 3], aie.engine = "S2MM0" }
  physical.stream_dma(%sA_0_b#1: !physical.istream<i32>) {
    %0 = physical.stream_dma_connect(%lA_0_b[0->1], %A_0_b[0:1024]: memref<1024xi32>, %0)
  } { aie.tile = [7, 3], aie.engine = "S2MM0" }
  physical.stream_dma(%sA_1_a#1: !physical.istream<i32>) {
    %0 = physical.stream_dma_connect(%lA_1_a[0->1], %A_1_a[0:1024]: memref<1024xi32>, %0)
  } { aie.tile = [6, 4], aie.engine = "S2MM0" }
  physical.stream_dma(%sA_1_b#1: !physical.istream<i32>) {
    %0 = physical.stream_dma_connect(%lA_1_b[0->1], %A_1_b[0:1024]: memref<1024xi32>, %0)
  } { aie.tile = [7, 4], aie.engine = "S2MM0" }

  physical.stream_dma(%sB_0#0: !physical.ostream<i32>) {
    %0 = physical.stream_dma_connect<2>(%leB_0_0[1->0], %eB_0_0[0:1024]: memref<1024xi32>, %1)
    %1 = physical.stream_dma_connect<3>(%leB_0_1[1->0], %eB_0_1[0:1024]: memref<1024xi32>, %0)
  } { aie.tile = [6, 0], aie.engine = "MM2S1" }
  physical.stream_dma(%sB_1#0: !physical.ostream<i32>) {
    %0 = physical.stream_dma_connect<4>(%leB_1_0[1->0], %eB_1_0[0:1024]: memref<1024xi32>, %1)
    %1 = physical.stream_dma_connect<5>(%leB_1_1[1->0], %eB_1_1[0:1024]: memref<1024xi32>, %0)
  } { aie.tile = [7, 0], aie.engine = "MM2S0" }
  physical.stream_dma(%sB_0_0#1: !physical.istream<i32>) {
    %0 = physical.stream_dma_connect(%lB_0_0[0->1], %B_0_0[0:1024]: memref<1024xi32>, %0)
  } { aie.tile = [6, 3], aie.engine = "S2MM1" }
  physical.stream_dma(%sB_0_1#1: !physical.istream<i32>) {
    %0 = physical.stream_dma_connect(%lB_0_1[0->1], %B_0_1[0:1024]: memref<1024xi32>, %0)
  } { aie.tile = [6, 4], aie.engine = "S2MM1" }
  physical.stream_dma(%sB_1_0#1: !physical.istream<i32>) {
    %0 = physical.stream_dma_connect(%lB_1_0[0->1], %B_1_0[0:1024]: memref<1024xi32>, %0)
  } { aie.tile = [7, 3], aie.engine = "S2MM1" }
  physical.stream_dma(%sB_1_1#1: !physical.istream<i32>) {
    %0 = physical.stream_dma_connect(%lB_1_1[0->1], %B_1_1[0:1024]: memref<1024xi32>, %0)
  } { aie.tile = [7, 4], aie.engine = "S2MM1" }

  physical.stream_dma(%sC_0#0: !physical.ostream<i32>) {
    %0 = physical.stream_dma_connect<6>(%lC_0[1->0], %C_0[0:1024]: memref<1024xi32>, %0)
  } { aie.tile = [6, 4], aie.engine = "MM2S0" }
  physical.stream_dma(%sC_1#0: !physical.ostream<i32>) {
    %0 = physical.stream_dma_connect<7>(%lC_1[1->0], %C_1[0:1024]: memref<1024xi32>, %0)
  } { aie.tile = [7, 4], aie.engine = "MM2S0" }
  physical.stream_dma(%seC_0#1: !physical.istream<i32>) {
    %0 = physical.stream_dma_connect(%leC_0[0->1], %eC_0[0:1024]: memref<1024xi32>, %0)
  } { aie.tile = [7, 0], aie.engine = "S2MM0" }
  physical.stream_dma(%seC_1#1: !physical.istream<i32>) {
    %1 = physical.stream_dma_connect(%leC_1[0->1], %eC_1[0:1024]: memref<1024xi32>, %1)
  } { aie.tile = [7, 0], aie.engine = "S2MM1" }

  physical.stream_hub(
          %sA#1,
        %sB_0#1,   %sB_1#1,
        %sC_0#1,   %sC_1#1,
      %sA_0_a#0, %sA_0_b#0, %sA_1_a#0, %sA_1_b#0,
      %sB_0_0#0, %sB_0_1#0, %sB_1_0#0, %sB_1_1#0,
       %seC_0#0,  %seC_1#0)
    { aie.impl = "broadcast_packet" }
    : (!physical.istream<i32>,
       !physical.istream<i32>, !physical.istream<i32>,
       !physical.istream<i32>, !physical.istream<i32>,
       !physical.ostream<i32>, !physical.ostream<i32>, !physical.ostream<i32>, !physical.ostream<i32>,
       !physical.ostream<i32>, !physical.ostream<i32>, !physical.ostream<i32>, !physical.ostream<i32>,
       !physical.ostream<i32>, !physical.ostream<i32>)
    -> !physical.stream_hub<i32>

  func.func private @extern_kernel(%A: memref<1024xi32>, %B: memref<1024xi32>, %acc: memref<1024xi32>, %C: memref<1024xi32>) -> ()

  func.func private @kernel(%A: memref<1024xi32>, %lA: !physical.lock,
                   %B: memref<1024xi32>, %lB: !physical.lock,
                   %acc: memref<1024xi32>, %lacc: !physical.lock,
                   %C: memref<1024xi32>, %lC: !physical.lock) {
    cf.br ^bb
^bb:
    physical.lock_acquire<1>(%lA)
    physical.lock_acquire<1>(%lB)
    physical.lock_acquire<1>(%lacc)
    physical.lock_acquire<0>(%lC)
    func.call @extern_kernel(%A, %B, %acc, %C) : (memref<1024xi32>, memref<1024xi32>, memref<1024xi32>, memref<1024xi32>) -> ()
    physical.lock_release<1>(%lC)
    physical.lock_release<0>(%lacc)
    physical.lock_release<0>(%lB)
    physical.lock_release<0>(%lA)
    cf.br ^bb
  }

  physical.core @kernel(%A_0_a, %lA_0_a, %B_0_0, %lB_0_0, %S_0_0, %lS_0_0, %S_0_1, %lS_0_1)
    { aie.tile = [6, 3] }: (memref<1024xi32>, !physical.lock, memref<1024xi32>, !physical.lock, memref<1024xi32>, !physical.lock, memref<1024xi32>, !physical.lock) -> !physical.core

  physical.core @kernel(%A_1_a, %lA_1_a, %B_0_1, %lB_0_1, %S_0_1, %lS_0_1,   %C_0,   %lC_0)
    { aie.tile = [6, 4] }: (memref<1024xi32>, !physical.lock, memref<1024xi32>, !physical.lock, memref<1024xi32>, !physical.lock, memref<1024xi32>, !physical.lock) -> !physical.core

  physical.core @kernel(%A_0_b, %lA_0_b, %B_1_0, %lB_1_0, %S_1_0, %lS_1_0, %S_1_1, %lS_1_1)
    { aie.tile = [7, 3] }: (memref<1024xi32>, !physical.lock, memref<1024xi32>, !physical.lock, memref<1024xi32>, !physical.lock, memref<1024xi32>, !physical.lock) -> !physical.core

  physical.core @kernel(%A_1_b, %lA_1_b, %B_1_1, %lB_1_1, %S_1_1, %lS_1_1,   %C_1,   %lC_1)
    { aie.tile = [7, 4] }: (memref<1024xi32>, !physical.lock, memref<1024xi32>, !physical.lock, memref<1024xi32>, !physical.lock, memref<1024xi32>, !physical.lock) -> !physical.core

}
