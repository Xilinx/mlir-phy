// RUN: phy-opt %s | FileCheck %s

// CHECK: physical.stream
%stream = physical.stream(): !physical.stream<i32>

// CHECK: physical.istream
%input = physical.istream(%stream: !physical.stream<i32>)

// CHECK: physical.ostream
%output = physical.ostream(%stream: !physical.stream<i32>)

// CHECK: physical.stream
%stream_tag = physical.stream<[1, 2]>(): !physical.stream<i32>
