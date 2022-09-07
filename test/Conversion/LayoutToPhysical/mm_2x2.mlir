// RUN: phy-opt --convert-layout-to-physical="device=aie" %s | FileCheck %s

// CHECK-LABEL: module @MM_2x2
module @MM_2x2 {

  %QA_0   = spatial.queue<1>(): !spatial.queue<memref<1024xi32>>
  %QA_1   = spatial.queue<1>(): !spatial.queue<memref<1024xi32>>
  %QB_0_0 = spatial.queue<1>(): !spatial.queue<memref<1024xi32>>
  %QB_0_1 = spatial.queue<1>(): !spatial.queue<memref<1024xi32>>
  %QB_1_0 = spatial.queue<1>(): !spatial.queue<memref<1024xi32>>
  %QB_1_1 = spatial.queue<1>(): !spatial.queue<memref<1024xi32>>
  %QC_0   = spatial.queue<1>(): !spatial.queue<memref<1024xi32>>
  %QC_1   = spatial.queue<1>(): !spatial.queue<memref<1024xi32>>

  %S_0_0  = spatial.queue<1>(): !spatial.queue<memref<1024xi32>>
  %S_0_1  = spatial.queue<1>(): !spatial.queue<memref<1024xi32>>
  %S_1_0  = spatial.queue<1>(): !spatial.queue<memref<1024xi32>>
  %S_1_1  = spatial.queue<1>(): !spatial.queue<memref<1024xi32>>

  func.func private @extern_kernel(%A: memref<1024xi32>, %B: memref<1024xi32>, %acc: memref<1024xi32>, %C: memref<1024xi32>) -> ()
  func.func private @kernel(%A: !spatial.queue<memref<1024xi32>>, %B: !spatial.queue<memref<1024xi32>>, %S: !spatial.queue<memref<1024xi32>>, %C: !spatial.queue<memref<1024xi32>>) {
    cf.br ^bb
^bb:
    %aA = spatial.front(%A): memref<1024xi32>
    %aB = spatial.front(%B): memref<1024xi32>
    %aS = spatial.front(%S): memref<1024xi32>
    %aC = spatial.emplace(%C): memref<1024xi32>
    func.call @extern_kernel(%aA, %aB, %aS, %aC) : (memref<1024xi32>, memref<1024xi32>, memref<1024xi32>, memref<1024xi32>) -> ()
    spatial.pop(%A: !spatial.queue<memref<1024xi32>>)
    spatial.pop(%B: !spatial.queue<memref<1024xi32>>)
    spatial.pop(%S: !spatial.queue<memref<1024xi32>>)
    spatial.push(%C: !spatial.queue<memref<1024xi32>>)
    cf.br ^bb
  }

  %node_0_0 = spatial.node @kernel(%QA_0, %QB_0_0, %S_0_0, %S_0_1): (!spatial.queue<memref<1024xi32>>, !spatial.queue<memref<1024xi32>>, !spatial.queue<memref<1024xi32>>, !spatial.queue<memref<1024xi32>>) -> !spatial.node
  %node_0_1 = spatial.node @kernel(%QA_1, %QB_0_1, %S_0_1, %QC_0) : (!spatial.queue<memref<1024xi32>>, !spatial.queue<memref<1024xi32>>, !spatial.queue<memref<1024xi32>>, !spatial.queue<memref<1024xi32>>) -> !spatial.node
  %node_1_0 = spatial.node @kernel(%QA_0, %QB_1_0, %S_1_0, %S_1_1): (!spatial.queue<memref<1024xi32>>, !spatial.queue<memref<1024xi32>>, !spatial.queue<memref<1024xi32>>, !spatial.queue<memref<1024xi32>>) -> !spatial.node
  %node_1_1 = spatial.node @kernel(%QA_1, %QB_1_1, %S_1_1, %QC_1) : (!spatial.queue<memref<1024xi32>>, !spatial.queue<memref<1024xi32>>, !spatial.queue<memref<1024xi32>>, !spatial.queue<memref<1024xi32>>) -> !spatial.node

  layout.platform<"vck190"> {
    layout.device<"aie"> {

      // TODO: move to layout.device<"global_memory">
      layout.place<"external_address/2203318222848/buffer,tile/6.0/id/0/lock">(%QA_0: !spatial.queue<memref<1024xi32>>)
      layout.place<"external_address/2203318226944/buffer,tile/6.0/id/1/lock">(%QA_1: !spatial.queue<memref<1024xi32>>)
      layout.place<"external_address/2203318231040/buffer,tile/6.0/id/2/lock">(%QB_0_0: !spatial.queue<memref<1024xi32>>)
      layout.place<"external_address/2203318235136/buffer,tile/6.0/id/3/lock">(%QB_0_1: !spatial.queue<memref<1024xi32>>)
      layout.place<"external_address/2203318239232/buffer,tile/7.0/id/0/lock">(%QB_1_0: !spatial.queue<memref<1024xi32>>)
      layout.place<"external_address/2203318243328/buffer,tile/7.0/id/1/lock">(%QB_1_1: !spatial.queue<memref<1024xi32>>)
      layout.place<"external_address/2203318247424/buffer,tile/7.0/id/2/lock">(%QC_0: !spatial.queue<memref<1024xi32>>)
      layout.place<"external_address/2203318251520/buffer,tile/7.0/id/3/lock">(%QC_1: !spatial.queue<memref<1024xi32>>)
      
      layout.place<"tile/6.3/id/2/buffer,tile/6.3/id/2/lock">(%S_0_0: !spatial.queue<memref<1024xi32>>)
      layout.place<"tile/6.3/id/3/buffer,tile/6.3/id/3/lock">(%S_0_1: !spatial.queue<memref<1024xi32>>)
      layout.place<"tile/7.3/id/2/buffer,tile/7.3/id/2/lock">(%S_1_0: !spatial.queue<memref<1024xi32>>)
      layout.place<"tile/7.3/id/3/buffer,tile/7.3/id/3/lock">(%S_1_1: !spatial.queue<memref<1024xi32>>)
      layout.place<"tile/6.3/core">(%node_0_0: !spatial.node)
      layout.place<"tile/6.4/core">(%node_0_1: !spatial.node)
      layout.place<"tile/7.3/core">(%node_1_0: !spatial.node)
      layout.place<"tile/7.4/core">(%node_1_1: !spatial.node)

      // Direct accesses
      layout.route<[]>(%S_0_0: !spatial.queue<memref<1024xi32>> -> %node_0_0: !spatial.node)
      layout.route<[]>(%node_0_0: !spatial.node -> %S_0_1: !spatial.queue<memref<1024xi32>>)
      layout.route<[]>(%S_0_1: !spatial.queue<memref<1024xi32>> -> %node_0_1: !spatial.node)
      
      // Direct accesses
      layout.route<[]>(%S_1_0: !spatial.queue<memref<1024xi32>> -> %node_1_0: !spatial.node)
      layout.route<[]>(%node_1_0: !spatial.node -> %S_1_1: !spatial.queue<memref<1024xi32>>)
      layout.route<[]>(%S_1_1: !spatial.queue<memref<1024xi32>> -> %node_1_1: !spatial.node)

      layout.route<["tile/6.0/engine/MM2S/id/0/stream_dma", "tile/6.0/port/DMA.O/id/0/stream", "impl/broadcast_packet/stream_hub",
                    "tile/6.3/port/DMA.I/id/0/stream", "tile/6.3/engine/S2MM/id/0/stream_dma", "tile/6.3/id/0/buffer,tile/6.3/id/0/lock"]>
                    (%QA_0: !spatial.queue<memref<1024xi32>> -> %node_0_0: !spatial.node)
      layout.route<["tile/6.0/engine/MM2S/id/0/stream_dma", "tile/6.0/port/DMA.O/id/0/stream", "impl/broadcast_packet/stream_hub",
                    "tile/6.4/port/DMA.I/id/0/stream", "tile/6.4/engine/S2MM/id/0/stream_dma", "tile/6.4/id/0/buffer,tile/6.4/id/0/lock"]>
                    (%QA_1: !spatial.queue<memref<1024xi32>> -> %node_0_1: !spatial.node)
      layout.route<["tile/6.0/engine/MM2S/id/0/stream_dma", "tile/6.0/port/DMA.O/id/0/stream", "impl/broadcast_packet/stream_hub",
                    "tile/7.3/port/DMA.I/id/0/stream", "tile/7.3/engine/S2MM/id/0/stream_dma", "tile/7.3/id/0/buffer,tile/7.3/id/0/lock"]>
                    (%QA_0: !spatial.queue<memref<1024xi32>> -> %node_1_0: !spatial.node)
      layout.route<["tile/6.0/engine/MM2S/id/0/stream_dma", "tile/6.0/port/DMA.O/id/0/stream", "impl/broadcast_packet/stream_hub",
                    "tile/7.4/port/DMA.I/id/0/stream", "tile/7.4/engine/S2MM/id/0/stream_dma", "tile/7.4/id/0/buffer,tile/7.4/id/0/lock"]>
                    (%QA_1: !spatial.queue<memref<1024xi32>> -> %node_1_1: !spatial.node)

      layout.route<["tile/6.0/engine/MM2S/id/1/stream_dma", "tile/6.0/port/DMA.O/id/1/stream", "impl/broadcast_packet/stream_hub",
                    "tile/6.3/port/DMA.I/id/1/stream", "tile/6.3/engine/S2MM/id/1/stream_dma", "tile/6.3/id/1/buffer,tile/6.3/id/1/lock"]>
                    (%QB_0_0: !spatial.queue<memref<1024xi32>> -> %node_0_0: !spatial.node)
      layout.route<["tile/6.0/engine/MM2S/id/1/stream_dma", "tile/6.0/port/DMA.O/id/1/stream", "impl/broadcast_packet/stream_hub",
                    "tile/6.4/port/DMA.I/id/1/stream", "tile/6.4/engine/S2MM/id/1/stream_dma", "tile/6.4/id/1/buffer,tile/6.4/id/1/lock"]>
                    (%QB_0_1: !spatial.queue<memref<1024xi32>> -> %node_0_1: !spatial.node)

      layout.route<["tile/7.0/engine/MM2S/id/0/stream_dma", "tile/7.0/port/DMA.O/id/0/stream", "impl/broadcast_packet/stream_hub",
                    "tile/7.3/port/DMA.I/id/1/stream", "tile/7.3/engine/S2MM/id/1/stream_dma", "tile/7.3/id/1/buffer,tile/7.3/id/1/lock"]>
                    (%QB_1_0: !spatial.queue<memref<1024xi32>> -> %node_1_0: !spatial.node)
      layout.route<["tile/7.0/engine/MM2S/id/0/stream_dma", "tile/7.0/port/DMA.O/id/0/stream", "impl/broadcast_packet/stream_hub",
                    "tile/7.4/port/DMA.I/id/1/stream", "tile/7.4/engine/S2MM/id/1/stream_dma", "tile/7.4/id/1/buffer,tile/7.4/id/1/lock"]>
                    (%QB_1_1: !spatial.queue<memref<1024xi32>> -> %node_1_1: !spatial.node)

      layout.route<["tile/6.4/id/2/buffer,tile/6.4/id/2/lock", "tile/6.4/engine/MM2S/id/0/stream_dma", "tile/6.4/port/DMA.O/id/0/stream",
                    "impl/broadcast_packet/stream_hub", "tile/7.0/port/DMA.I/id/0/stream", "tile/7.0/engine/S2MM/id/0/stream_dma"]>
                    (%node_0_1: !spatial.node -> %QC_0: !spatial.queue<memref<1024xi32>>)
      layout.route<["tile/7.4/id/2/buffer,tile/7.4/id/2/lock", "tile/7.4/engine/MM2S/id/0/stream_dma", "tile/7.4/port/DMA.O/id/0/stream",
                    "impl/broadcast_packet/stream_hub", "tile/7.0/port/DMA.I/id/1/stream", "tile/7.0/engine/S2MM/id/1/stream_dma"]>
                    (%node_1_1: !spatial.node -> %QC_1: !spatial.queue<memref<1024xi32>>)
    }
  }
}
