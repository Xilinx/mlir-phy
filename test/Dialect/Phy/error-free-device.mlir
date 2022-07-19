// RUN: (phy-opt %s 2>&1 || true) | FileCheck %s

// CHECK: 'phy.device' op expects parent op 'phy.platform'
phy.device<"hls"> {}
