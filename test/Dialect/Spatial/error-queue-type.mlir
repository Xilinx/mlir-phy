// RUN: (phy-opt %s 2>&1 || true) | FileCheck %s
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// CHECK-LABEL: 'spatial.queue' op result #0 must be a queue
%queue = spatial.queue<2>(): !spatial.queue<i32>
