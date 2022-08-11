//===- BusOps.td -------------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_DIALECT_PHYSICAL_BUS_OPS_TD
#define MLIR_PHY_DIALECT_PHYSICAL_BUS_OPS_TD

include "phy/Dialect/Physical/PhysicalBase.td"

//===----------------------------------------------------------------------===//
// BusOp
//===----------------------------------------------------------------------===//

def BusOp : Physical_Op<"bus"> {
  let summary = "addressed bus creation operation";
  let description = [{
    The `physical.bus` operation represents a creation of an addressed bus
    that can have buffers mapped to its memory space using the 'physical.mmap'
    operation.

    Example:

    ```mlir
    %buf = physical.buffer() : memref<1024xi32>
    %bus = physical.bus() : !physical.bus<i32>
    physical.mmap(%bus[10:15], %buf[20:] : memref<1024xi32>)
    %pe = physical.core @func(%bus) : (!physical.bus<i32>) -> !physical.core
    ```
  }];

  let arguments = (ins);
  let results = (outs BusTypeOf<AnyType>:$bus);

  let assemblyFormat = [{
    `(` `)` attr-dict `:` type($bus)
  }];
}

//===----------------------------------------------------------------------===//
// MmapOp
//===----------------------------------------------------------------------===//

def MmapOp : Physical_Op<"mmap",
    [TypesMatchWith<"type of 'bus' matches element type of the 'buffer'",
                    "buffer", "bus",
                    "BusType::get($_self.getContext(), "
                      "$_self.cast<MemRefType>().getElementType())">]> {

  let summary = "creation of a buffer into an addressed bus memory space";
  let description = [{
    An operation that maps the 'buffer' starting the 'offset'-th element,
    into the 'bus'.  The mapped address is from 'begin'-th element
    (inclusive), to the 'end'-th element (exclusive) on the bus.

    Example:

    ```mlir
    physical.mmap(%abus[10:15], %buf[20:] : memref<1024xi32>)
    // abus[10] will be buf[20], abus[11] will be buf[21], ...
    ```
  }];

  let arguments = (ins BusTypeOf<AnyType>:$bus,
                       I32Attr:$begin,
                       I32Attr:$end,
                       AnyStaticShapeMemRef:$buffer,
                       I32Attr:$offset);

  let assemblyFormat = [{
    `(` $bus `[` $begin `:` $end `]` `,` $buffer `[` $offset `:` `]`
    `:` type($buffer) `)` attr-dict 
  }];
}

//===----------------------------------------------------------------------===//
// StartLoadOp
//===----------------------------------------------------------------------===//

def StartLoadOp : Physical_Op<"start_load",
    [TypesMatchWith<"result's type is the handle of element type of 'memref'",
                    "memref", "result",
                    "HandleType::get($_self.getContext(), "
                      "($_self.isa<MemRefType>() ? ("
                        "$_self.cast<MemRefType>().getElementType()"
                      ") : ("
                        "$_self.cast<BusType>().getDatatype()"
                      ")))">,
     MemRefsNormalizable]> {

  let summary = "non-blocking load of the data";
  let description = [{
    A non-blocking bus access that reads the data from a buffer, or an addressed
    bus, as specified in 'memref'.  This operation returns a handle, which may be
    waited using `physical.wait` to get the access result.

    Example:

    ```mlir
    %load_handle = physical.start_load %mem[%idx] : memref<1024xi32>
    %0 = physical.wait(%load_handle) : i32
    ```
  }];

  let arguments = (ins AnyLoadStoreType:$memref,
                       Variadic<Index>:$indices);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $memref `[` $indices `]` attr-dict `:` type($memref)
  }];
}

//===----------------------------------------------------------------------===//
// StartStoreOp
//===----------------------------------------------------------------------===//

def StartStoreOp : Physical_Op<"start_store",
    [TypesMatchWith<"type of 'value' matches element type of 'memref'",
                    "memref", "value",
                    "$_self.isa<MemRefType>() ? ("
                        "$_self.cast<MemRefType>().getElementType()"
                    ") : ("
                        "$_self.cast<BusType>().getDatatype()"
                    ")">,
     TypesMatchWith<"result's type is a handle of NoneType",
                    "memref", "handle",
                    "HandleType::get($_self.getContext(), "
                      "NoneType::get($_self.getContext()))">,
     MemRefsNormalizable]> {

  let summary = "non-blocking store of the data";
  let description = [{
    A non-blocking bus access that stores the data to a buffer, or an addressed
    bus, as specified in 'memref'.  This operation returns a handle, which may
    be waited using `physical.wait`.

    Example:

    ```mlir
    %store_handle = physical.start_store %0, %mem[%idx] : memref<1024xi32>
    physical.wait(%store_handle) : none
    ```
  }];

  let arguments = (ins AnyType:$value,
                       AnyLoadStoreType:$memref,
                       Variadic<Index>:$indices);
  let results = (outs HandleTypeOf<NoneType>:$handle);

  let assemblyFormat = [{
    $value `,` $memref `[` $indices `]` attr-dict `:` type($memref)
  }];
}

//===----------------------------------------------------------------------===//
// WaitOp
//===----------------------------------------------------------------------===//

def WaitOp : Physical_Op<"wait",
    [TypesMatchWith<"handle's data type matches result's type",
                    "result", "handle",
                    "HandleType::get($_self.getContext(), $_self)">]> {

  let summary = "blocking wait until a handle is ready";
  let description = [{
    A non-blocking bus access returns a handle, which may be waited using this
    operation to get the access result.  For store access, none is returned.

    Example:

    ```mlir
    %0 = physical.wait(%load_handle) : i32
    physical.wait(%store_handle) : none
    ```
  }];

  let arguments = (ins AnyHandleType:$handle);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    `(` $handle `)` attr-dict `:` type($result) 
  }];
}

#endif  // MLIR_PHY_DIALECT_PHYSICAL_BUS_OPS_TD
