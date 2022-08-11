//===- PhysicalTypes.td ------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_DIALECT_PHYSICAL_TYPES_TD
#define MLIR_PHY_DIALECT_PHYSICAL_TYPES_TD

include "phy/Dialect/Physical/PhysicalBase.td"

include "phy/Dialect/Physical/Types/BasicTypes.td"
include "phy/Dialect/Physical/Types/BusTypes.td"
include "phy/Dialect/Physical/Types/CacheTypes.td"
include "phy/Dialect/Physical/Types/FifoTypes.td"

#endif  // MLIR_PHY_DIALECT_PHYSICAL_TYPES_TD
