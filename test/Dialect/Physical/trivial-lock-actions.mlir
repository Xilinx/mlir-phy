// RUN: phy-opt %s | FileCheck %s

// CHECK: physical.lock
%lock = physical.lock<0>()

// CHECK: physical.lock_acquire
physical.lock_acquire<0>(%lock)

// CHECK: physical.lock_release
physical.lock_release<1>(%lock)