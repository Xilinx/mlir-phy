// RUN: phy-opt %s | FileCheck %s

// CHECK-LABEL: func.func @func
func.func @func(%buf:  !spatial.queue<memref<1024xi32>>,
                %fifo: !spatial.queue<memref<i32>>) {
  func.return
}
// CHECK: spatial.queue
%buf    = spatial.queue<1>() : !spatial.queue<memref<1024xi32>>
// CHECK: spatial.queue
%fifo1  = spatial.queue<2>() : !spatial.queue<memref<i32>>
// CHECK: spatial.queue
%fifo2  = spatial.queue<2>() : !spatial.queue<memref<i32>>
// CHECK: spatial.bridge
%bridge = spatial.bridge(%fifo1 -> %fifo2: !spatial.queue<memref<i32>>)
// CHECK: spatial.node
%node   = spatial.node @func(%buf, %fifo1)
        : (!spatial.queue<memref<1024xi32>>, !spatial.queue<memref<i32>>)
        -> !spatial.node

// CHECK: spatial.flow
%flow1  = spatial.flow(%buf:    !spatial.queue<memref<1024xi32>>
                    -> %node:   !spatial.node)
        : !spatial.flow<memref<1024xi32>>

// CHECK: spatial.flow
%flow2  = spatial.flow(%node:   !spatial.node
                    -> %fifo1:  !spatial.queue<memref<i32>>)
        : !spatial.flow<memref<i32>>

// CHECK: spatial.flow
%flow3  = spatial.flow(%fifo1:  !spatial.queue<memref<i32>>
                    -> %bridge: !spatial.node)
        : !spatial.flow<memref<i32>>
        
// CHECK: spatial.flow
%flow4  = spatial.flow(%bridge: !spatial.node
                    -> %fifo2:  !spatial.queue<memref<i32>>)
        : !spatial.flow<memref<i32>>

// CHECK: layout.platform<"versal">
layout.platform<"versal"> {
  // CHECK: layout.device<"pl">
  layout.device<"pl"> {
    // CHECK: layout.place<["slr0"]>
    layout.place<["slr0"]>(%buf: !spatial.queue<memref<1024xi32>>)
    // CHECK: layout.route<["slr0-slr1"]>
    layout.route<["slr0-slr1"]>(%flow1: !spatial.flow<memref<1024xi32>>)
    // CHECK: layout.place<["slr1"]>
    layout.place<["slr1"]>(%node: !spatial.node)
    // CHECK: layout.route<[]>
    layout.route<[]>(%flow2: !spatial.flow<memref<i32>>)
    // CHECK: layout.place<["slr1"]>
    layout.place<["slr1"]>(%fifo1: !spatial.queue<memref<i32>>)
  }
  // CHECK: layout.device<"aie">
  layout.device<"aie"> {
    // CHECK: layout.place<["shimswitchbox"]>
    layout.place<["shimswitchbox"]>(%bridge: !spatial.node)
    // CHECK: layout.route<["tile70-tile71", "tile71-tile72"]>
    layout.route<["tile70-tile71", "tile71-tile72"]>(
      %flow4 : !spatial.flow<memref<i32>>)
    // CHECK: layout.place<["tile72"]>
    layout.place<["tile72"]>(%fifo2: !spatial.queue<memref<i32>>)
  }
  // CHECK: layout.route<["pl-aie"]>
  layout.route<["pl-aie"]>(%flow3: !spatial.flow<memref<i32>>)
}
