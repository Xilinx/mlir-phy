//===- PhysicalOps.td --------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_DIALECT_PHYSICAL_OPS_TD
#define MLIR_PHY_DIALECT_PHYSICAL_OPS_TD

include "phy/Dialect/Physical/PhysicalBase.td"

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/SymbolInterfaces.td"

class Physical_Op<string mnemonic, list<Trait> traits = []> :
    Op<Physical_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// CoreOp
//===----------------------------------------------------------------------===//

def CoreOp : Physical_Op<"core",
    [CallOpInterface, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {

  let summary = "processing core creation operation";
  let description = [{
    The `physical.core` operation represents a creation of a processing core
    that has the function argument as its entry point.  The processing core
    will be free-running and the function will be invoked.  The function must
    be within the same symbol scope as the operation.  The operands must match
    the specified function type. The function is encoded as a symbol reference
    attribute named `callee`.

    Example:

    ```mlir
    %core = physical.core @my_adder(%buf)
          : (memref<1024xi32>) -> !physical.core
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee,
                   Variadic<AnyCoreArgsType>:$operands);
  let results = (outs CoreType:$core);

  let extraClassDeclaration = [{
    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    ::mlir::CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<::mlir::SymbolRefAttr>("callee");
    }
  }];

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, $core)
  }];
}

//===----------------------------------------------------------------------===//
// BufferOp
//===----------------------------------------------------------------------===//

def BufferOp : Physical_Op<"buffer"> {
  let summary = "buffer creation operation";
  let description = [{
    The `physical.bufferfer` operation represents a creation of a buffer that has
    the type argument as its datatype.  A buffer is a memory space that stores
    data.  A buffer can be randomly accessed.  It can have a device-specific
    attribute of location.

    Example:

    ```mlir
    %buffer = physical.bufferfer() : memref<1024xi32>
    ```
  }];

  let results = (outs AnyStaticShapeMemRef:$buffer);

  let assemblyFormat = [{
    `(` `)` attr-dict `:` type($buffer)
  }];
}

//===----------------------------------------------------------------------===//
// CacheOp
//===----------------------------------------------------------------------===//

def CacheOp : Physical_Op<"cache",
    [TypesMatchWith<"type of 'upstream' matches element type of the cache",
                    "cache", "upstream",
                    "BusType::get($_self.getContext(), "
                      "$_self.cast<CacheType>().getDatatype())">,
     TypesMatchWith<"type of 'downstream' matches element type of the cache",
                    "cache", "downstream",
                    "BusType::get($_self.getContext(), "
                      "$_self.cast<CacheType>().getDatatype())">]> {

  let summary = "creation of a cache layer between two addressed buses";
  let description = [{
    An operation creating a cache to connect two buses.  With the cache,
    the memory access on the 'downstream' bus from the buffers on the 'upstream'
    bus will be cached, and the 'downstream' bus is able to receive data from
    the buffers on the 'upstream' bus.  The 'physical.cache' type specifies the
    number of elements that can be cached.

    Example:

    ```mlir
    %bus1 = physical.bus() : !physical.bus<i32>
    %bus2 = physical.bus() : !physical.bus<i32>
    %cache = physical.cache(%bus1, %bus2) : !physical.cache<i32, 1024>
    ```
  }];

  let arguments = (ins BusTypeOf<AnyType>:$upstream,
                       BusTypeOf<AnyType>:$downstream);
  let results = (outs CacheTypeOf<AnyType>:$cache);

  let assemblyFormat = [{
    `(` $upstream `,` $downstream `)` attr-dict `:` type($cache)
  }];
}

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

  let arguments = (ins HandleTypeOf<AnyType>:$handle);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    `(` $handle `)` attr-dict `:` type($result) 
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

#endif  // MLIR_PHY_DIALECT_PHYSICAL_OPS_TD
