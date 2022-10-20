// RUN: (phy-opt %s 2>&1 || true) | FileCheck %s
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

%bus1 = physical.bus(): !physical.bus<i32>
%bus2 = physical.bus(): !physical.bus<i32>

// CHECK: expects different type than prior uses: '!physical.bus<i16>' vs '!physical.bus<i32>'
%cache = physical.bus_cache(%bus1, %bus2) : !physical.bus_cache<i16, 1024>