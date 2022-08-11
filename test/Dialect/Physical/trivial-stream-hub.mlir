// RUN: phy-opt %s | FileCheck %s

// CHECK: physical.stream
%stream1 = physical.stream(): !physical.stream<i32>

// CHECK: physical.istream
%input = physical.istream(%stream1: !physical.stream<i32>)

// CHECK: physical.stream
%stream2 = physical.stream(): !physical.stream<i32>

// CHECK: physical.ostream
%output = physical.ostream(%stream2: !physical.stream<i32>)

// CHECK: physical.stream_hub
%hub = physical.stream_hub(%input, %output)
     : (!physical.istream<i32>, !physical.ostream<i32>)
     -> !physical.stream_hub<i32>