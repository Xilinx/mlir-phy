// RUN: phy-opt --convert-layout-to-physical="device=aie" %s | FileCheck %s

func.func private @kernel() {
  cf.br ^bb
^bb:
  cf.br ^bb
}

%node = spatial.node @kernel(): () -> !spatial.node

layout.platform<"vck190"> {
  layout.device<"aie"> {
    
    // CHECK: physical.core @kernel() {aie.tile = "6.3"} : () -> !physical.core
    layout.place<"tile/6.3/core">(%node: !spatial.node)

  }
}
