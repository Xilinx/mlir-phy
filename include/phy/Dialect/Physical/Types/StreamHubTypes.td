//===- StreamHubTypes.td -----------------------------------*- tablegen -*-===//
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_DIALECT_PHYSICAL_STREAM_HUB_TYPES_TD
#define MLIR_PHY_DIALECT_PHYSICAL_STREAM_HUB_TYPES_TD

include "phy/Dialect/Physical/PhysicalBase.td"

//===----------------------------------------------------------------------===//
// StreamHubType
//===----------------------------------------------------------------------===//

def StreamHubType : Physical_Type<"StreamHub", "stream_hub"> {
  let summary = "A type specifiying a stream hub for broadcasting";
  let description = [{
    A stream hub receives data from input streams and broadcast to output
    stream.  The supported endpoint count of a stream hub is target-dependent.
    If the data in a stream is optionally tagged for packet switching, only
    the streams accepting the tags will receive the data.
  }];

  let parameters = (ins "Type":$datatype);
  let assemblyFormat = "`<` $datatype `>`";
}

class StreamHubTypeOf<Type element> : Type<
  And<[
    CPred<"$_self.isa<phy::physical::StreamHubType>()">,
    SubstLeaves<
      "$_self",
      "$_self.cast<phy::physical::StreamHubType>().getDatatype()",
      element.predicate>
  ]>,
  "a stream hub of " # element.summary
>;

def AnyStreamHubType : Type<
  Or<[
    StreamHubTypeOf<AnyType>.predicate,
  ]>,
  "a stream hub"
>;

#endif  // MLIR_PHY_DIALECT_PHYSICAL_ROUTER_TYPES_TD
