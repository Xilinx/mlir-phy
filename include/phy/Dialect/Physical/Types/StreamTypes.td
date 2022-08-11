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
  let summary = "A type specifiying a streaming connection";
  let description = [{
    A stream connects two endpoints and provides a streaming connection.  It
    does not provide inflight data buffering, which is instead implemented as
    a `physical.stream_fifo` or `physical.buffer`s and `physical.stream_dma`s.
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
// IStreamEndpointType
//===----------------------------------------------------------------------===//

def IStreamEndpointType : Physical_Type<"IStreamEndpoint", "istream"> {
  let summary = "A type specifiying a streaming endpoint for input";
  let description = [{
    A stream endpoint that takes data as input.
  }];

  let parameters = (ins "Type":$datatype);
  let assemblyFormat = "`<` $datatype `>`";
}

class IStreamEndpointTypeOf<Type element> : Type<
  And<[
    CPred<"$_self.isa<phy::physical::IStreamEndpointType>()">,
    SubstLeaves<
      "$_self",
      "$_self.cast<phy::physical::IStreamEndpointType>().getDatatype()",
      element.predicate>
  ]>,
  "an input endpoint of a stream of " # element.summary
>;

def AnyIStreamEndpointType : Type<
  Or<[
    IStreamEndpointTypeOf<AnyType>.predicate,
  ]>,
  "an input endpoint of a stream"
>;

//===----------------------------------------------------------------------===//
// OStreamEndpointType
//===----------------------------------------------------------------------===//

def OStreamEndpointType : Physical_Type<"OStreamEndpoint", "ostream"> {
  let summary = "A type specifiying a streaming endpoint for output";
  let description = [{
    A stream endpoint that outputs data.
  }];

  let parameters = (ins "Type":$datatype);
  let assemblyFormat = "`<` $datatype `>`";
}

class OStreamEndpointTypeOf<Type element> : Type<
  And<[
    CPred<"$_self.isa<phy::physical::OStreamEndpointType>()">,
    SubstLeaves<
      "$_self",
      "$_self.cast<phy::physical::OStreamEndpointType>().getDatatype()",
      element.predicate>
  ]>,
  "an output endpoint of a stream of " # element.summary
>;

def AnyOStreamEndpointType : Type<
  Or<[
    OStreamEndpointTypeOf<AnyType>.predicate,
  ]>,
  "an output endpoint of a stream"
>;

//===----------------------------------------------------------------------===//
// Type Constraints
//===----------------------------------------------------------------------===//

def AnyStreamEndpointType : Type<
  Or<[
    AnyIStreamEndpointType.predicate,
    AnyOStreamEndpointType.predicate,
  ]>,
  "a stream endpoint"
>;

#endif  // MLIR_PHY_DIALECT_PHYSICAL_STREAM_TYPES_TD
