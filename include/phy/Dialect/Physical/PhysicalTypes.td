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

include "mlir/IR/AttrTypeBase.td"

class Physical_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Physical_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// CoreType
//===----------------------------------------------------------------------===//

def CoreType : Physical_Type<"Core", "core"> {
  let summary = "A type specifiying a processing core";
  let description = [{
    A core is a logical function that computes.  It can have a device-specific
    attribute of location, e.g. a core in CPU or a tile core in AIE.
  }];
}

//===----------------------------------------------------------------------===//
// BusType
//===----------------------------------------------------------------------===//

def BusType : Physical_Type<"Bus", "bus"> {
  let summary = "A type specifiying a bus with address mapping specification";
  let description = [{
    A bus with its address space used by buffers.  In the defining operation, a
    buffer can specify how its memory address space is mapped to the bus.  A
    bus can be used in the same way as a 'memref' in a PE with the
    'physical.start_load' and 'physical.start_store' operations.
  }];

  let parameters = (ins "Type":$datatype);
  let assemblyFormat = "`<` $datatype `>`";
}

//===----------------------------------------------------------------------===//
// FifoType
//===----------------------------------------------------------------------===//

def FifoType : Physical_Type<"Fifo", "fifo"> {
  let summary = "A streaming first-in first-out storage unit";
  let description = [{
    Fifo are first-in first-out storage elements, which takes an element in
    a basic data type as input and buffers it.  The elements in the fifo is
    sent to the output whenever they are ready.  A fifo can have only one input
    connection and one output connection.
  }];

  let parameters = (ins "Type":$datatype);
  let assemblyFormat = "`<` $datatype `>`";
}

//===----------------------------------------------------------------------===//
// CacheType
//===----------------------------------------------------------------------===//

def CacheType : Physical_Type<"Cache", "cache"> {
  let summary = "A type specifiying a cache connecting two buses";
  let description = [{
    A bus can be connected to an upstream bus using caches that provides caching.
    The parameter 'size' specifies the elements to be stored in the cache.
  }];

  let parameters = (ins "Type":$datatype, "int":$size);
  let assemblyFormat = "`<` $datatype `,` $size `>`";
}

//===----------------------------------------------------------------------===//
// HandleType
//===----------------------------------------------------------------------===//

def HandleType : Physical_Type<"Handle", "handle"> {
  let summary = "A type specifiying a on-going memory access";
  let description = [{
    A non-blocking memory access returns a handle, which may be waited to get
    the access result.
  }];

  let parameters = (ins "Type":$datatype);
  let assemblyFormat = "`<` $datatype `>`";
}

#endif  // MLIR_PHY_DIALECT_PHYSICAL_TYPES_TD
