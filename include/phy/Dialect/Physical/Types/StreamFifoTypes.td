//===- FifoTypes.td ----------------------------------------*- tablegen -*-===//
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_DIALECT_PHYSICAL_STREAM_FIFO_TYPES_TD
#define MLIR_PHY_DIALECT_PHYSICAL_STREAM_FIFO_TYPES_TD

include "phy/Dialect/Physical/PhysicalBase.td"

//===----------------------------------------------------------------------===//
// StreamFifoType
//===----------------------------------------------------------------------===//

def StreamFifoType : Physical_Type<"StreamFifo", "stream_fifo"> {
  let summary = "A streaming first-in first-out storage unit";
  let description = [{
    Fifos are first-in first-out storage elements, which takes an element in
    as input and buffers it.  The elements in the fifo is sent to the output
    whenever they are ready.  A fifo can have only one input connection and one
    output connection.  The tag of the stream data will be preseved.
  }];

  let parameters = (ins "Type":$datatype);
  let assemblyFormat = "`<` $datatype `>`";
}

class StreamFifoTypeOf<Type element> : Type<
  And<[
    CPred<"$_self.isa<phy::physical::StreamFifoType>()">,
    SubstLeaves<
      "$_self",
      "$_self.cast<phy::physical::StreamFifoType>().getDatatype()",
      element.predicate>
  ]>,
  "a stream fifo of " # element.summary
>;

def AnyStreamFifoType : Type<
  Or<[
    StreamFifoTypeOf<AnyType>.predicate,
  ]>,
  "a stream fifo fifo"
>;

#endif  // MLIR_PHY_DIALECT_PHYSICAL_STREAM_FIFO_TYPES_TD
