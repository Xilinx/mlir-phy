// RUN: (phy-opt %s 2>&1 || true) | FileCheck %s

func.func @function() {
  func.return
}
%pe = phy.pe @function() : () -> !phy.pe
phy.platform<"xilinx"> {

  // CHECK: 'phy.place' op expects parent op 'phy.device'
  phy.place<"slr0">(%pe : !phy.pe)
}
