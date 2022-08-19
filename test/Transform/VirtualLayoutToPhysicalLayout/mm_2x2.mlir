// RUN: phy-opt %s | FileCheck %s

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
    layout.device<"global_memory"> {
      layout.place<"affinity/6.0/depth/1/locked_buffer">(%QA_0: !spatial.queue<memref<1024xi32>>)
      layout.place<"affinity/6.0/depth/1/locked_buffer">(%QA_1: !spatial.queue<memref<1024xi32>>)
      layout.place<"affinity/6.0/depth/1/locked_buffer">(%QB_0_0: !spatial.queue<memref<1024xi32>>)
      layout.place<"affinity/6.0/depth/1/locked_buffer">(%QB_0_1: !spatial.queue<memref<1024xi32>>)
      layout.place<"affinity/7.0/depth/1/locked_buffer">(%QB_1_0: !spatial.queue<memref<1024xi32>>)
      layout.place<"affinity/7.0/depth/1/locked_buffer">(%QB_1_1: !spatial.queue<memref<1024xi32>>)
      layout.place<"affinity/7.0/depth/1/locked_buffer">(%QC_0: !spatial.queue<memref<1024xi32>>)
      layout.place<"affinity/7.0/depth/1/locked_buffer">(%QC_1: !spatial.queue<memref<1024xi32>>)
    }
    layout.device<"aie"> {
      layout.place<"affinity/6.3/locked_buffer">(%S_0_0: !spatial.queue<memref<1024xi32>>)
      layout.place<"affinity/6.3/locked_buffer">(%S_0_1: !spatial.queue<memref<1024xi32>>)
      layout.place<"affinity/7.3/locked_buffer">(%S_1_0: !spatial.queue<memref<1024xi32>>)
      layout.place<"affinity/7.3/locked_buffer">(%S_1_1: !spatial.queue<memref<1024xi32>>)
      layout.place<"tile/6.3/core">(%node_0_0: !spatial.node)
      layout.place<"tile/6.4/core">(%node_0_1: !spatial.node)
      layout.place<"tile/7.3/core">(%node_1_0: !spatial.node)
      layout.place<"tile/7.4/core">(%node_1_1: !spatial.node)

      layout.route<["tile/6.0/buffer_to_stream", "impl/broadcast_packet/stream_hub",
                    "tile/6.3/stream_to_buffer", "affinity/6.3/depth/1/locked_buffer"]>
                    (%QA_0: !spatial.queue<memref<1024xi32>> -> %node_0_0: !spatial.node)
      layout.route<["tile/6.0/buffer_to_stream", "impl/broadcast_packet/stream_hub",
                    "tile/6.4/stream_to_buffer", "affinity/6.4/depth/1/locked_buffer"]>
                    (%QA_1: !spatial.queue<memref<1024xi32>> -> %node_0_1: !spatial.node)
      layout.route<["tile/6.0/buffer_to_stream", "impl/broadcast_packet/stream_hub",
                    "tile/7.3/stream_to_buffer", "affinity/7.3/depth/1/locked_buffer"]>
                    (%QA_0: !spatial.queue<memref<1024xi32>> -> %node_1_0: !spatial.node)
      layout.route<["tile/6.0/buffer_to_stream", "impl/broadcast_packet/stream_hub",
                    "tile/7.4/stream_to_buffer", "affinity/7.4/depth/1/locked_buffer"]>
                    (%QA_1: !spatial.queue<memref<1024xi32>> -> %node_1_1: !spatial.node)
      // TODO: hints to interleave Q_0 and Q_1 so that tile/6.0/buffer_to_stream can be overcommitted
      // this hint shall be generated by static analysis finding that pushing QA_0 always follows QA_1

      layout.route<["tile/6.0/buffer_to_stream", "impl/broadcast_packet/stream_hub",
                    "tile/6.3/stream_to_buffer", "affinity/6.3/depth/1/locked_buffer"]>
                    (%QB_0_0: !spatial.queue<memref<1024xi32>> -> %node_0_0: !spatial.node)
      layout.route<["tile/6.0/buffer_to_stream", "impl/broadcast_packet/stream_hub",
                    "tile/6.4/stream_to_buffer", "affinity/6.4/depth/1/locked_buffer"]>
                    (%QB_0_1: !spatial.queue<memref<1024xi32>> -> %node_0_1: !spatial.node)
      layout.route<["tile/7.0/buffer_to_stream", "impl/broadcast_packet/stream_hub",
                    "tile/7.3/stream_to_buffer", "affinity/7.3/depth/1/locked_buffer"]>
                    (%QB_1_0: !spatial.queue<memref<1024xi32>> -> %node_1_0: !spatial.node)
      layout.route<["tile/7.0/buffer_to_stream", "impl/broadcast_packet/stream_hub",
                    "tile/7.4/stream_to_buffer", "affinity/7.4/depth/1/locked_buffer"]>
                    (%QB_1_1: !spatial.queue<memref<1024xi32>> -> %node_1_1: !spatial.node)

      layout.route<["affinity/6.4/depth/1/locked_buffer", "tile/6.4/buffer_to_stream",
                    "impl/broadcast_packet/stream_hub", "tile/7.0/stream_to_buffer"]>
                    (%node_0_1: !spatial.node -> %QC_0: !spatial.queue<memref<1024xi32>>)
      layout.route<["affinity/7.4/depth/1/locked_buffer", "tile/7.4/buffer_to_stream",
                    "impl/broadcast_packet/stream_hub", "tile/7.0/stream_to_buffer"]>
                    (%node_1_1: !spatial.node -> %QC_1: !spatial.queue<memref<1024xi32>>)
    }
  }
}
