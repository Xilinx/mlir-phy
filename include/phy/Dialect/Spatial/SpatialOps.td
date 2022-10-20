//===- SpatialOps.td ---------------------------------------*- tablegen -*-===//
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_DIALECT_SPATIAL_OPS_TD
#define MLIR_PHY_DIALECT_SPATIAL_OPS_TD

include "phy/Dialect/Spatial/SpatialBase.td"

include "phy/Dialect/Spatial/Operations/FlowAccessOps.td"
include "phy/Dialect/Spatial/Operations/FlowControlOps.td"
include "phy/Dialect/Spatial/Operations/GraphOps.td"

#endif  // MLIR_PHY_DIALECT_SPATIAL_OPS_TD
