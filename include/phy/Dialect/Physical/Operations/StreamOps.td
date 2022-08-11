//===- StreamOps.td ----------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_DIALECT_PHYSICAL_STREAM_OPS_TD
#define MLIR_PHY_DIALECT_PHYSICAL_STREAM_OPS_TD

include "phy/Dialect/Physical/PhysicalBase.td"

//===----------------------------------------------------------------------===//
// StreamOp
//===----------------------------------------------------------------------===//

def StreamOp : Physical_Op<"stream"> {
  let summary = "streaming connection creation operation";
  let description = [{
    The `physical.stream` operation represents a creation of a stream that
    connects two endpoints and provides a streaming connection.  The created
    stream can be connected as an operand in `physical.core` operations for the
    software function to access and communicate, or by a `physical.stream_dma`
    operation.  Streams can be connected using `physical.stream_hub`s.

    Example:

    ```mlir
    %stream = physical.stream(): !physical.stream<i32>
    ```
  }];

  let results = (outs AnyStreamType:$stream);

  let assemblyFormat = [{
    `(` `)` attr-dict `:` type($stream)
  }];
}

//===----------------------------------------------------------------------===//
// IStreamOp
//===----------------------------------------------------------------------===//

def IStreamOp : Physical_Op<"istream",
    [TypesMatchWith<"endpoint's data type is the same as 'stream'",
                    "stream", "endpoint",
                    "IStreamEndpointType::get($_self.getContext(), "
                      "$_self.cast<StreamType>().getDatatype()"
                      ")">]> {
  let summary = "obtaining the input endpoint of the stream";
  let description = [{
    The `physical.istream` operation returns the input endpoint of the stream
    for usage as a `physical.core` operand, etc, to push data into the stream.

    Example:

    ```mlir
    %endpoint = physical.istream(%stream: !physical.stream<i32>)
    ```
  }];

  let arguments = (ins AnyStreamType:$stream);
  let results = (outs AnyIStreamEndpointType:$endpoint);

  let assemblyFormat = [{
    `(` $stream `:` type($stream) `)` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// OStreamOp
//===----------------------------------------------------------------------===//

def OStreamOp : Physical_Op<"ostream",
    [TypesMatchWith<"endpoint's data type is the same as 'stream'",
                    "stream", "endpoint",
                    "OStreamEndpointType::get($_self.getContext(), "
                      "$_self.cast<StreamType>().getDatatype()"
                      ")">]> {
  let summary = "obtaining the output endpoint of the stream";
  let description = [{
    The `physical.ostream` operation returns the output endpoint of the stream
    for usage as a `physical.core` operand, etc, to pop data from the stream.

    Example:

    ```mlir
    %endpoint = physical.ostream(%stream: !physical.stream<i32>)
    ```
  }];

  let arguments = (ins AnyStreamType:$stream);
  let results = (outs AnyOStreamEndpointType:$endpoint);

  let assemblyFormat = [{
    `(` $stream `:` type($stream) `)` attr-dict
  }];
}

#endif  // MLIR_PHY_DIALECT_PHYSICAL_STREAM_OPS_TD
