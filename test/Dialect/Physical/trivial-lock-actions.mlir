// RUN: phy-opt %s | FileCheck %s
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// CHECK: physical.lock
%lock = physical.lock<0>()

// CHECK: physical.lock_acquire
physical.lock_acquire<0>(%lock)

// CHECK: physical.lock_release
physical.lock_release<1>(%lock)