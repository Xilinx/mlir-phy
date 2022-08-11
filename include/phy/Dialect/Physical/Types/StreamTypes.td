//===- StreamTypes.td --------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_DIALECT_PHYSICAL_STREAM_TYPES_TD
#define MLIR_PHY_DIALECT_PHYSICAL_STREAM_TYPES_TD

include "phy/Dialect/Physical/PhysicalBase.td"

//===----------------------------------------------------------------------===//
// StreamType
//===----------------------------------------------------------------------===//

def StreamType : Physical_Type<"Stream", "stream"> {
  let summary = "A type specifiying a streaming connection of a basic type";
  let description = [{
    A stream connects two endpoints and provides a streaming connection of a
    basic datatype.  It does not provide inflight data buffering, which is
    instead implemented as a `physical.stream_fifo` or `physical.buffer`s and
    `physical.stream_dma`s.
  }];

  let parameters = (ins "Type":$datatype);
  let assemblyFormat = "`<` $datatype `>`";
}

class StreamTypeOf<Type element> : Type<
  And<[
    CPred<"$_self.isa<phy::physical::StreamType>()">,
    SubstLeaves<
      "$_self",
      "$_self.cast<phy::physical::StreamType>().getDatatype()",
      element.predicate>
  ]>,
  "a stream of " # element.summary
>;

def AnyStreamType : Type<
  Or<[
    StreamTypeOf<AnyType>.predicate,
  ]>,
  "a stream"
>;

//===----------------------------------------------------------------------===//
// StreamInEndpointType
//===----------------------------------------------------------------------===//

def StreamInEndpointType : Physical_Type<"StreamInEndpoint", "stream_in"> {
  let summary = "A type specifiying a streaming endpoint for input";
  let description = [{
    A stream endpoint that takes data of a basic datatype as input.
  }];

  let parameters = (ins "Type":$datatype);
  let assemblyFormat = "`<` $datatype `>`";
}

class StreamInEndpointTypeOf<Type element> : Type<
  And<[
    CPred<"$_self.isa<phy::physical::StreamInEndpointType>()">,
    SubstLeaves<
      "$_self",
      "$_self.cast<phy::physical::StreamInEndpointType>().getDatatype()",
      element.predicate>
  ]>,
  "an input endpoint of a stream of " # element.summary
>;

def AnyStreamInEndpointType : Type<
  Or<[
    StreamInEndpointTypeOf<AnyType>.predicate,
  ]>,
  "an input endpoint of a stream"
>;

//===----------------------------------------------------------------------===//
// StreamOutEndpointType
//===----------------------------------------------------------------------===//

def StreamOutEndpointType : Physical_Type<"StreamOutEndpoint", "stream_out"> {
  let summary = "A type specifiying a streaming endpoint for output";
  let description = [{
    A stream endpoint that outputs data of a basic datatype.
  }];

  let parameters = (ins "Type":$datatype);
  let assemblyFormat = "`<` $datatype `>`";
}

class StreamOutEndpointTypeOf<Type element> : Type<
  And<[
    CPred<"$_self.isa<phy::physical::StreamOutEndpointType>()">,
    SubstLeaves<
      "$_self",
      "$_self.cast<phy::physical::StreamOutEndpointType>().getDatatype()",
      element.predicate>
  ]>,
  "an output endpoint of a stream of " # element.summary
>;

def AnyStreamOutEndpointType : Type<
  Or<[
    StreamOutEndpointTypeOf<AnyType>.predicate,
  ]>,
  "an output endpoint of a stream"
>;

//===----------------------------------------------------------------------===//
// Type Constraints
//===----------------------------------------------------------------------===//

def AnyStreamEndpointType : Type<
  Or<[
    AnyStreamInEndpointType.predicate,
    AnyStreamOutEndpointType.predicate,
  ]>,
  "a stream endpoint"
>;

#endif  // MLIR_PHY_DIALECT_PHYSICAL_STREAM_TYPES_TD
