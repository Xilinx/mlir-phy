// RUN: (phy-opt %s 2>&1 || true) | FileCheck %s

func.func @function() {
  func.return
}
%node = spatial.node @function() : () -> !spatial.node
layout.platform<"xilinx"> {

  // CHECK: 'layout.place' op expects parent op 'layout.device'
  layout.place<"slr0">(%node: !spatial.node)
}
