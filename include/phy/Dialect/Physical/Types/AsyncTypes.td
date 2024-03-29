//===- AsyncTypes.td ---------------------------------------*- tablegen -*-===//
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_DIALECT_PHYSICAL_ASYNC_TYPES_TD
#define MLIR_PHY_DIALECT_PHYSICAL_ASYNC_TYPES_TD

include "phy/Dialect/Physical/PhysicalBase.td"
include "phy/Dialect/Physical/Types/BusTypes.td"

//===----------------------------------------------------------------------===//
// AsyncHandleType
//===----------------------------------------------------------------------===//

def AsyncHandleType : Physical_Type<"AsyncHandle", "async_handle"> {
  let summary = "A type specifiying a on-going memory access";
  let description = [{
    A non-blocking memory access returns a handle, which may be waited to get
    the access result.
  }];

  let parameters = (ins "Type":$datatype);
  let assemblyFormat = "`<` $datatype `>`";
}

class AsyncHandleTypeOf<Type element> : Type<
  And<[
    CPred<"$_self.isa<phy::physical::AsyncHandleType>()">,
    SubstLeaves<
      "$_self",
      "$_self.cast<phy::physical::AsyncHandleType>().getDatatype()",
      element.predicate>
  ]>,
  "an async handle of " # element.summary
>;

def AnyAsyncHandleType : Type<
  Or<[
    AsyncHandleTypeOf<AnyType>.predicate,
  ]>,
  "a handle"
>;

//===----------------------------------------------------------------------===//
// Type Constraints
//===----------------------------------------------------------------------===//

def AnyLoadStoreType : Type<
  Or<[
    AnyStaticShapeMemRef.predicate,
    AnyBusType.predicate,
  ]>,
  "a memref or a bus"
>;

#endif  // MLIR_PHY_DIALECT_PHYSICAL_ASYNC_TYPES_TD
