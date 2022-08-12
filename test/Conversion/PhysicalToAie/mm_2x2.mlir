// RUN: phy-opt %s | FileCheck %s

// CHECK-LABEL: module @MM_2x2
module @MM_2x2 {

  %LHS_tile0 = physical.buffer() { aie.external_address = 0x020100000000 }: memref<1024xi32>
  %LHS_tile1 = physical.buffer() { aie.external_address = 0x020100001000 }: memref<1024xi32>
  %RHS_tile0 = physical.buffer() { aie.external_address = 0x020100002000 }: memref<1024xi32>
  %RHS_tile1 = physical.buffer() { aie.external_address = 0x020100003000 }: memref<1024xi32>
  %RHS_tile2 = physical.buffer() { aie.external_address = 0x020100004000 }: memref<1024xi32>
  %RHS_tile3 = physical.buffer() { aie.external_address = 0x020100005000 }: memref<1024xi32>
  %Out_tile0 = physical.buffer() { aie.external_address = 0x020100006000 }: memref<1024xi32>
  %Out_tile1 = physical.buffer() { aie.external_address = 0x020100008000 }: memref<1024xi32>
  %buf63_0 = physical.buffer() { aie.tile = [6, 3] }: memref<1024xi32>
  %buf63_1 = physical.buffer() { aie.tile = [6, 3] }: memref<1024xi32>
  %buf63_2 = physical.buffer() { aie.tile = [6, 3] }: memref<1024xi32>
  %buf63_3 = physical.buffer() { aie.tile = [6, 3] }: memref<1024xi32>
  %buf64_0 = physical.buffer() { aie.tile = [6, 4] }: memref<1024xi32>
  %buf64_1 = physical.buffer() { aie.tile = [6, 4] }: memref<1024xi32>
  %buf64_2 = physical.buffer() { aie.tile = [6, 4] }: memref<1024xi32>
  %buf73_0 = physical.buffer() { aie.tile = [7, 3] }: memref<1024xi32>
  %buf73_1 = physical.buffer() { aie.tile = [7, 3] }: memref<1024xi32>
  %buf73_2 = physical.buffer() { aie.tile = [7, 3] }: memref<1024xi32>
  %buf73_3 = physical.buffer() { aie.tile = [7, 3] }: memref<1024xi32>
  %buf74_0 = physical.buffer() { aie.tile = [7, 4] }: memref<1024xi32>
  %buf74_1 = physical.buffer() { aie.tile = [7, 4] }: memref<1024xi32>
  %buf74_2 = physical.buffer() { aie.tile = [7, 4] }: memref<1024xi32>

  %lock_LHS_tile0 = physical.lock<0>() { aie.tile = [6, 0], aie.id = 0 }
  %lock_LHS_tile1 = physical.lock<0>() { aie.tile = [6, 0], aie.id = 1 }
  %lock_RHS_tile0 = physical.lock<0>() { aie.tile = [6, 0], aie.id = 2 }
  %lock_RHS_tile1 = physical.lock<0>() { aie.tile = [6, 0], aie.id = 3 }
  %lock_RHS_tile2 = physical.lock<0>() { aie.tile = [7, 0], aie.id = 0 }
  %lock_RHS_tile3 = physical.lock<0>() { aie.tile = [7, 0], aie.id = 1 }
  %lock_Out_tile0 = physical.lock<0>() { aie.tile = [7, 0], aie.id = 2 }
  %lock_Out_tile1 = physical.lock<0>() { aie.tile = [7, 0], aie.id = 3 }
  %lock63_0 = physical.lock<0>() { aie.tile = [6, 3], aie.id = 0 }
  %lock63_1 = physical.lock<0>() { aie.tile = [6, 3], aie.id = 1 }
  %lock63_2 = physical.lock<1>() { aie.tile = [6, 3], aie.id = 2 }
  %lock63_3 = physical.lock<0>() { aie.tile = [6, 3], aie.id = 3 }
  %lock64_0 = physical.lock<0>() { aie.tile = [6, 4], aie.id = 0 }
  %lock64_1 = physical.lock<0>() { aie.tile = [6, 4], aie.id = 1 }
  %lock64_2 = physical.lock<0>() { aie.tile = [6, 4], aie.id = 2 }
  %lock73_0 = physical.lock<0>() { aie.tile = [7, 3], aie.id = 0 }
  %lock73_1 = physical.lock<0>() { aie.tile = [7, 3], aie.id = 1 }
  %lock73_2 = physical.lock<1>() { aie.tile = [7, 3], aie.id = 2 }
  %lock73_3 = physical.lock<0>() { aie.tile = [7, 3], aie.id = 3 }
  %lock74_0 = physical.lock<0>() { aie.tile = [7, 4], aie.id = 0 }
  %lock74_1 = physical.lock<0>() { aie.tile = [7, 4], aie.id = 1 }
  %lock74_2 = physical.lock<0>() { aie.tile = [7, 4], aie.id = 2 }

  %stream60_mm2s0:2   = physical.stream<[0, 1]>(){ aie.tile = [6, 0], aie.port = "O.DMA.0" }: (!physical.ostream<i32>, !physical.istream<i32>)
  %stream60_mm2s1:2   = physical.stream<[2, 3]>(){ aie.tile = [6, 0], aie.port = "O.DMA.1" }: (!physical.ostream<i32>, !physical.istream<i32>)
  %stream70_mm2s0:2   = physical.stream<[4, 5]>(){ aie.tile = [7, 0], aie.port = "O.DMA.0" }: (!physical.ostream<i32>, !physical.istream<i32>)
  %stream70_s2mm0:2   = physical.stream<[6, 7]>(){ aie.tile = [7, 0], aie.port = "I.DMA.0" }: (!physical.ostream<i32>, !physical.istream<i32>)
  %stream63_s2mm0:2   = physical.stream<[0]>(){ aie.tile = [6, 3], aie.port = "I.DMA.0" }: (!physical.ostream<i32>, !physical.istream<i32>)
  %stream63_s2mm1:2   = physical.stream<[2]>(){ aie.tile = [6, 3], aie.port = "I.DMA.1" }: (!physical.ostream<i32>, !physical.istream<i32>)
  %stream64_s2mm0:2   = physical.stream<[1]>(){ aie.tile = [6, 4], aie.port = "I.DMA.0" }: (!physical.ostream<i32>, !physical.istream<i32>)
  %stream64_s2mm1:2   = physical.stream<[3]>(){ aie.tile = [6, 4], aie.port = "I.DMA.1" }: (!physical.ostream<i32>, !physical.istream<i32>)
  %stream64_mm2s0:2   = physical.stream<[6]>(){ aie.tile = [6, 4], aie.port = "O.DMA.0" }: (!physical.ostream<i32>, !physical.istream<i32>)
  %stream73_s2mm0:2   = physical.stream<[0]>(){ aie.tile = [7, 3], aie.port = "I.DMA.0" }: (!physical.ostream<i32>, !physical.istream<i32>)
  %stream73_s2mm1:2   = physical.stream<[4]>(){ aie.tile = [7, 3], aie.port = "I.DMA.1" }: (!physical.ostream<i32>, !physical.istream<i32>)
  %stream74_s2mm0:2   = physical.stream<[1]>(){ aie.tile = [7, 4], aie.port = "I.DMA.0" }: (!physical.ostream<i32>, !physical.istream<i32>)
  %stream74_s2mm1:2   = physical.stream<[5]>(){ aie.tile = [7, 4], aie.port = "I.DMA.1" }: (!physical.ostream<i32>, !physical.istream<i32>)
  %stream74_mm2s0:2   = physical.stream<[7]>(){ aie.tile = [7, 4], aie.port = "O.DMA.0" }: (!physical.ostream<i32>, !physical.istream<i32>)

  physical.stream_dma(%stream60_mm2s0#0: !physical.ostream<i32>) {
    %0 = physical.stream_dma_connect<0>(%lock_LHS_tile0[1->0], %LHS_tile0[0:1024]: memref<1024xi32>, %1)
    %1 = physical.stream_dma_connect<1>(%lock_LHS_tile1[1->0], %LHS_tile1[0:1024]: memref<1024xi32>, %0)
  } { aie.tile = [6, 0], aie.engine = "MM2S0" }

  physical.stream_dma(%stream60_mm2s1#0: !physical.ostream<i32>) {
    %0 = physical.stream_dma_connect<2>(%lock_RHS_tile0[1->0], %RHS_tile0[0:1024]: memref<1024xi32>, %1)
    %1 = physical.stream_dma_connect<3>(%lock_RHS_tile1[1->0], %RHS_tile1[0:1024]: memref<1024xi32>, %0)
  } { aie.tile = [6, 0], aie.engine = "MM2S1" }

  physical.stream_dma(%stream70_mm2s0#0: !physical.ostream<i32>) {
    %0 = physical.stream_dma_connect<4>(%lock_RHS_tile2[1->0], %RHS_tile2[0:1024]: memref<1024xi32>, %1)
    %1 = physical.stream_dma_connect<5>(%lock_RHS_tile3[1->0], %RHS_tile3[0:1024]: memref<1024xi32>, %0)
  } { aie.tile = [7, 0], aie.engine = "MM2S0" }

  physical.stream_dma(%stream70_s2mm0#1: !physical.istream<i32>) {
    %0 = physical.stream_dma_connect(%lock_Out_tile0[0->1], %Out_tile0[0:1024]: memref<1024xi32>, %1)
    %1 = physical.stream_dma_connect(%lock_Out_tile1[0->1], %Out_tile1[0:1024]: memref<1024xi32>, %0)
  } { aie.tile = [7, 0], aie.engine = "S2MM0" }

  physical.stream_dma(%stream63_s2mm0#1: !physical.istream<i32>) {
    %0 = physical.stream_dma_connect(%lock63_0[0->1], %buf63_0[0:1024]: memref<1024xi32>, %0)
  } { aie.tile = [6, 3], aie.engine = "S2MM0" }

  physical.stream_dma(%stream63_s2mm1#1: !physical.istream<i32>) {
    %0 = physical.stream_dma_connect(%lock63_1[0->1], %buf63_1[0:1024]: memref<1024xi32>, %0)
  } { aie.tile = [6, 3], aie.engine = "S2MM1" }

  physical.stream_dma(%stream64_s2mm0#1: !physical.istream<i32>) {
    %0 = physical.stream_dma_connect(%lock64_0[0->1], %buf64_0[0:1024]: memref<1024xi32>, %0)
  } { aie.tile = [6, 4], aie.engine = "S2MM0" }

  physical.stream_dma(%stream64_s2mm1#1: !physical.istream<i32>) {
    %0 = physical.stream_dma_connect(%lock64_1[0->1], %buf64_1[0:1024]: memref<1024xi32>, %0)
  } { aie.tile = [6, 4], aie.engine = "S2MM1" }

  physical.stream_dma(%stream64_mm2s0#0: !physical.ostream<i32>) {
    %0 = physical.stream_dma_connect<6>(%lock64_2[1->0], %buf64_2[0:1024]: memref<1024xi32>, %0)
  } { aie.tile = [6, 4], aie.engine = "MM2S0" }

  physical.stream_dma(%stream73_s2mm0#1: !physical.istream<i32>) {
    %0 = physical.stream_dma_connect(%lock73_0[0->1], %buf73_0[0:1024]: memref<1024xi32>, %0)
  } { aie.tile = [7, 3], aie.engine = "S2MM0" }

  physical.stream_dma(%stream73_s2mm1#1: !physical.istream<i32>) {
    %0 = physical.stream_dma_connect(%lock73_1[0->1], %buf73_1[0:1024]: memref<1024xi32>, %0)
  } { aie.tile = [7, 3], aie.engine = "S2MM1" }

  physical.stream_dma(%stream74_s2mm0#1: !physical.istream<i32>) {
    %0 = physical.stream_dma_connect(%lock74_0[0->1], %buf74_0[0:1024]: memref<1024xi32>, %0)
  } { aie.tile = [7, 4], aie.engine = "S2MM0" }

  physical.stream_dma(%stream74_s2mm1#1: !physical.istream<i32>) {
    %0 = physical.stream_dma_connect(%lock74_1[0->1], %buf74_1[0:1024]: memref<1024xi32>, %0)
  } { aie.tile = [7, 4], aie.engine = "S2MM1" }

  physical.stream_dma(%stream74_mm2s0#0: !physical.ostream<i32>) {
    %0 = physical.stream_dma_connect<7>(%lock74_2[1->0], %buf74_2[0:1024]: memref<1024xi32>, %0)
  } { aie.tile = [7, 4], aie.engine = "MM2S0" }

  physical.stream_hub(
      %stream60_mm2s0#1, %stream60_mm2s0#1, %stream70_mm2s0#1, %stream64_mm2s0#1, %stream74_mm2s0#1,
      %stream70_s2mm0#0, %stream63_s2mm0#0, %stream63_s2mm1#0, %stream64_s2mm0#0, %stream64_s2mm1#0,
      %stream73_s2mm0#0, %stream73_s2mm1#0, %stream74_s2mm0#0, %stream74_s2mm1#0)
    { aie.impl = "broadcast_packet" }
    : (!physical.istream<i32>, !physical.istream<i32>, !physical.istream<i32>, !physical.istream<i32>, !physical.istream<i32>,
       !physical.ostream<i32>, !physical.ostream<i32>, !physical.ostream<i32>, !physical.ostream<i32>, !physical.ostream<i32>,
       !physical.ostream<i32>, !physical.ostream<i32>, !physical.ostream<i32>, !physical.ostream<i32>)
    -> !physical.stream_hub<i32>

  func.func private @extern_kernel(%A: memref<1024xi32>, %B: memref<1024xi32>, %acc: memref<1024xi32>, %C: memref<1024xi32>) -> ()

  func.func private @kernel(%A: memref<1024xi32>, %lock_A: !physical.lock,
                   %B: memref<1024xi32>, %lock_B: !physical.lock,
                   %acc: memref<1024xi32>, %lock_acc: !physical.lock,
                   %C: memref<1024xi32>, %lock_C: !physical.lock) {
    cf.br ^bb
^bb:
    physical.lock_acquire<1>(%lock_A)
    physical.lock_acquire<1>(%lock_B)
    physical.lock_acquire<1>(%lock_acc)
    physical.lock_acquire<0>(%lock_C)
    func.call @extern_kernel(%A, %B, %acc, %C) : (memref<1024xi32>, memref<1024xi32>, memref<1024xi32>, memref<1024xi32>) -> ()
    physical.lock_release<1>(%lock_C)
    physical.lock_release<0>(%lock_acc)
    physical.lock_release<0>(%lock_B)
    physical.lock_release<0>(%lock_A)
    cf.br ^bb
  }

  physical.core @kernel(%buf63_0, %lock63_0, %buf63_1, %lock63_1, %buf63_2, %lock63_2, %buf63_3, %lock63_3)
    { aie.tile = [6, 3] }: (memref<1024xi32>, !physical.lock, memref<1024xi32>, !physical.lock, memref<1024xi32>, !physical.lock, memref<1024xi32>, !physical.lock) -> !physical.core

  physical.core @kernel(%buf64_0, %lock64_0, %buf64_1, %lock64_1, %buf63_3, %lock63_3, %buf64_2, %lock64_2)
    { aie.tile = [6, 4] }: (memref<1024xi32>, !physical.lock, memref<1024xi32>, !physical.lock, memref<1024xi32>, !physical.lock, memref<1024xi32>, !physical.lock) -> !physical.core

  physical.core @kernel(%buf73_0, %lock73_0, %buf73_1, %lock73_1, %buf73_2, %lock73_2, %buf73_3, %lock73_3)
    { aie.tile = [7, 3] }: (memref<1024xi32>, !physical.lock, memref<1024xi32>, !physical.lock, memref<1024xi32>, !physical.lock, memref<1024xi32>, !physical.lock) -> !physical.core

  physical.core @kernel(%buf74_0, %lock74_0, %buf74_1, %lock74_1, %buf73_3, %lock73_3, %buf74_2, %lock74_2)
    { aie.tile = [7, 4] }: (memref<1024xi32>, !physical.lock, memref<1024xi32>, !physical.lock, memref<1024xi32>, !physical.lock, memref<1024xi32>, !physical.lock) -> !physical.core

}