// REQUIRES: aie_found
// RUN: phy-opt --convert-physical-to-aie %s | FileCheck %s
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// CHECK: %[[Tile:.*]] = AIE.tile(6, 0)

// CHECK: %[[Lock:.*]] = AIE.lock(%[[Tile]], 0)
// CHECK: AIE.useLock(%[[Lock]], Release, 1)
%0 = physical.lock<1>() { aie.tile = "6.0", aie.id = "0" }
