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
// StreamInOp
//===----------------------------------------------------------------------===//

def StreamInOp : Physical_Op<"stream_in",
    [TypesMatchWith<"endpoint's data type is the same as 'stream'",
                    "stream", "endpoint",
                    "StreamInEndpointType::get($_self.getContext(), "
                      "$_self.cast<StreamType>().getDatatype()"
                      ")">]> {
  let summary = "obtaining the input endpoint of the stream";
  let description = [{
    The `physical.stream_in` operation returns the input endpoint of the stream
    for usage as a `physical.core` operand, etc, to push data into the stream.

    Example:

    ```mlir
    %endpoint = physical.stream_in(%stream: !physical.stream<i32>)
    ```
  }];

  let arguments = (ins AnyStreamType:$stream);
  let results = (outs AnyStreamInEndpointType:$endpoint);

  let assemblyFormat = [{
    `(` $stream `:` type($stream) `)` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// StreamOutOp
//===----------------------------------------------------------------------===//

def StreamOutOp : Physical_Op<"stream_out",
    [TypesMatchWith<"endpoint's data type is the same as 'stream'",
                    "stream", "endpoint",
                    "StreamOutEndpointType::get($_self.getContext(), "
                      "$_self.cast<StreamType>().getDatatype()"
                      ")">]> {
  let summary = "obtaining the output endpoint of the stream";
  let description = [{
    The `physical.stream_out` operation returns the output endpoint of the stream
    for usage as a `physical.core` operand, etc, to pop data from the stream.

    Example:

    ```mlir
    %endpoint = physical.stream_out(%stream: !physical.stream<i32>)
    ```
  }];

  let arguments = (ins AnyStreamType:$stream);
  let results = (outs AnyStreamOutEndpointType:$endpoint);

  let assemblyFormat = [{
    `(` $stream `:` type($stream) `)` attr-dict
  }];
}

#endif  // MLIR_PHY_DIALECT_PHYSICAL_STREAM_OPS_TD
