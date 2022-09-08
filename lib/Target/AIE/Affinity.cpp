//===- Affinity.cpp ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "phy/Target/AIE/TargetResources.h"

#include <utility>

using namespace phy::connectivity;
using namespace phy::target;
using namespace phy::target::aie;

// Affinity code from Xilinx/mlir-aie/lib/AIEDialect.cpp,
// licensed under the Apache License v2.0 with LLVM Exceptions.
// (c) Copyright 2019 Xilinx Inc.

static bool isInternal(int src_col, int src_row, int dst_col, int dst_row) {
  return ((src_col == dst_col) && (src_row == dst_row));
}

static bool isWest(int src_col, int src_row, int dst_col, int dst_row) {
  return ((src_col == dst_col + 1) && (src_row == dst_row));
}

static bool isNorth(int src_col, int src_row, int dst_col, int dst_row) {
  return ((src_col == dst_col) && (src_row == dst_row - 1));
}

bool TargetResources::isLegalAffinity(int core_col, int core_row, int buf_col,
                                      int buf_row) {
  bool isLegalCore = (core_col > 0 && core_col <= array_width) &&
                     (core_row > 0 && core_row <= array_height);
  bool isLegalBuf = (buf_col > 0 && buf_col <= array_width) &&
                    (buf_row > 0 && buf_row <= array_height);

  bool IsEven_row = ((core_row % 2) == 0);

  bool IsMemWest =
      (isWest(core_col, core_row, buf_col, buf_row) && !IsEven_row) ||
      (isInternal(core_col, core_row, buf_col, buf_row) && IsEven_row);
  bool IsMemEast =
      (isWest(buf_col, buf_row, core_col, core_row) && IsEven_row) ||
      (isInternal(core_col, core_row, buf_col, buf_row) && !IsEven_row);
  bool IsMemNorth = isNorth(core_col, core_row, buf_col, buf_row);
  bool IsMemSouth = isNorth(buf_col, buf_row, core_col, core_row);

  return isLegalCore && isLegalBuf &&
         (IsMemSouth || IsMemNorth || IsMemWest || IsMemEast);
}

// End Xilinx code.

bool TargetResources::isShimTile(int col, int row) { return row == 0; }

std::set<std::pair<int, int>>
TargetResources::getAffinity(int col, int row, std::string neigh_type) {
  std::list<std::pair<int, int>> neighbors = {
      {-1, 0}, {1, 0}, {0, -1}, {0, 1}, {0, 0}};

  std::set<std::pair<int, int>> tiles;
  for (auto neighbor : neighbors) {
    int neigh_col = col + neighbor.first;
    int neigh_row = row + neighbor.second;

    if ((neigh_type == std::string("core") &&
         isLegalAffinity(neigh_col, neigh_row, col, row)) ||
        (neigh_type == std::string("buffer") &&
         isLegalAffinity(col, row, neigh_col, neigh_row))) {
      tiles.insert({neigh_col, neigh_row});
    }
  }

  return tiles;
}
