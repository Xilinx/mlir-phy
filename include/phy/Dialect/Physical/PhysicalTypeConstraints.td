//===- PhysicalTypeConstraints.td ---------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_DIALECT_PHYSICAL_TYPE_CONSTRAINTS_TD
#define MLIR_PHY_DIALECT_PHYSICAL_TYPE_CONSTRAINTS_TD

include "phy/Dialect/Physical/PhysicalBase.td"
include "phy/Dialect/Physical/PhysicalTypes.td"

class BusTypeOf<Type element> : Type<
  And<[
    CPred<"$_self.isa<BusType>()">,
    SubstLeaves<
      "$_self",
      "$_self.cast<BusType>().getDatatype()",
      element.predicate>
  ]>,
  "a bus of " # element.summary
>;

class HandleTypeOf<Type element> : Type<
  And<[
    CPred<"$_self.isa<HandleType>()">,
    SubstLeaves<
      "$_self",
      "$_self.cast<HandleType>().getDatatype()",
      element.predicate>
  ]>,
  "a handle of " # element.summary
>;

class CacheTypeOf<Type element> : Type<
  And<[
    CPred<"$_self.isa<CacheType>()">,
    SubstLeaves<
      "$_self",
      "$_self.cast<CacheType>().getDatatype()",
      element.predicate>
  ]>,
  "a cache of " # element.summary
>;

def AnyCoreArgsType : Type<
  Or<[
    AnyStaticShapeMemRef.predicate,
    BusTypeOf<AnyType>.predicate,
  ]>,
  "a memref or a bus"
>;

def AnyLoadStoreType : Type<
  Or<[
    AnyStaticShapeMemRef.predicate,
    BusTypeOf<AnyType>.predicate,
  ]>,
  "a memref or a bus"
>;

#endif  // MLIR_PHY_DIALECT_PHYSICAL_TYPE_CONSTRAINTS_TD
