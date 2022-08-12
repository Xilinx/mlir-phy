// RUN: phy-opt %s | FileCheck %s

// CHECK: physical.stream
%stream:2 = physical.stream(): (!physical.ostream<i32>, !physical.istream<i32>)

// CHECK: physical.stream
%stream_tag:2 = physical.stream<[1]>(): (!physical.ostream<i32>, !physical.istream<i32>)
