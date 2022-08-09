// RUN: (phy-opt %s 2>&1 || true) | FileCheck %s

// CHECK-LABEL: 'spatial.queue' op result #0 must be a queue
%queue = spatial.queue<2>(): !spatial.queue<i32>
