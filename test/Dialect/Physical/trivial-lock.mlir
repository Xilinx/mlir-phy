// RUN: phy-opt %s | FileCheck %s

// CHECK: physical.lock
%lock = physical.lock<0>()
