// RUN: phy-opt %s | FileCheck %s

// CHECK: physical.stream
%stream1:2 = physical.stream(): (!physical.ostream<i32>, !physical.istream<i32>)

// CHECK: physical.stream
%stream2:2 = physical.stream(): (!physical.ostream<i32>, !physical.istream<i32>)

// CHECK: physical.stream_hub
%hub = physical.stream_hub(%stream1#1, %stream2#0)
     : (!physical.istream<i32>, !physical.ostream<i32>)
     -> !physical.stream_hub<i32>