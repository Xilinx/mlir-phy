//===- PhysicalTypes.td ------------------------------------*- tablegen -*-===//
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_DIALECT_PHYSICAL_TYPES_TD
#define MLIR_PHY_DIALECT_PHYSICAL_TYPES_TD

include "phy/Dialect/Physical/PhysicalBase.td"

include "phy/Dialect/Physical/Types/AsyncTypes.td"
include "phy/Dialect/Physical/Types/BusCacheTypes.td"
include "phy/Dialect/Physical/Types/BusTypes.td"
include "phy/Dialect/Physical/Types/CoreTypes.td"
include "phy/Dialect/Physical/Types/LockTypes.td"
include "phy/Dialect/Physical/Types/StreamDmaTypes.td"
include "phy/Dialect/Physical/Types/StreamFifoTypes.td"
include "phy/Dialect/Physical/Types/StreamTypes.td"
include "phy/Dialect/Physical/Types/StreamHubTypes.td"

#endif  // MLIR_PHY_DIALECT_PHYSICAL_TYPES_TD
