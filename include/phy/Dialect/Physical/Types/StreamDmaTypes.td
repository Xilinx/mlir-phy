//===- StreamDmaTypes.td -----------------------------------*- tablegen -*-===//
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_DIALECT_PHYSICAL_STREAM_DMA_TYPES_TD
#define MLIR_PHY_DIALECT_PHYSICAL_STREAM_DMA_TYPES_TD

include "phy/Dialect/Physical/PhysicalBase.td"

//===----------------------------------------------------------------------===//
// StreamDmaConnectType
//===----------------------------------------------------------------------===//

def StreamDmaConnectType : Physical_Type<"StreamDmaConnect",
                                         "stream_dma_connect"> {
 
  let summary = "A stream DMA connection";
  let description = [{
    A stream DMA connection connects a buffer with a stream and perform DMA
    operations between them.  A variable in this type can be passed as an
    argument to another DMA connection as its successor.
  }];
}

#endif  // MLIR_PHY_DIALECT_PHYSICAL_STREAM_DMA_TYPES_TD
