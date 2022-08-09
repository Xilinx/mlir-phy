// RUN: (phy-opt %s 2>&1 || true) | FileCheck %s

// CHECK: 'layout.device' op expects parent op 'layout.platform'
layout.device<"hls"> {}
