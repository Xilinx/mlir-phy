// RUN: phy-opt %s | FileCheck %s

// CHECK: physical.stream
%stream = physical.stream(): !physical.stream<i32>

// CHECK: physical.istream
%input = physical.istream(%stream: !physical.stream<i32>)

// CHECK: physical.ostream
%output = physical.ostream(%stream: !physical.stream<i32>)
