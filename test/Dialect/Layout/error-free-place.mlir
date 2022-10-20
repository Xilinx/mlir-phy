// RUN: (phy-opt %s 2>&1 || true) | FileCheck %s
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

func.func @function() {
  func.return
}
%node = spatial.node @function() : () -> !spatial.node
layout.platform<"xilinx"> {

  // CHECK: 'layout.place' op expects parent op 'layout.device'
  layout.place<"slr0">(%node: !spatial.node)
}
