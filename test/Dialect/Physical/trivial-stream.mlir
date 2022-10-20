// RUN: phy-opt %s | FileCheck %s
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// CHECK: physical.stream
%stream:2 = physical.stream(): (!physical.ostream<i32>, !physical.istream<i32>)

// CHECK: physical.stream
%stream_tag:2 = physical.stream<[1]>(): (!physical.ostream<i32>, !physical.istream<i32>)
