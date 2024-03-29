//===- BusCacheTypes.td ------------------------------------*- tablegen -*-===//
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_DIALECT_PHYSICAL_BUS_CACHE_TYPES_TD
#define MLIR_PHY_DIALECT_PHYSICAL_BUS_CACHE_TYPES_TD

include "phy/Dialect/Physical/PhysicalBase.td"

//===----------------------------------------------------------------------===//
// BusCacheType
//===----------------------------------------------------------------------===//

def BusCacheType : Physical_Type<"BusCache", "bus_cache"> {
  let summary = "A type specifiying a cache connecting two buses";
  let description = [{
    A bus can be connected to an upstream bus using caches that provides caching.
    The parameter 'size' specifies the elements to be stored in the cache.
  }];

  let parameters = (ins "Type":$datatype, "int":$size);
  let assemblyFormat = "`<` $datatype `,` $size `>`";
}

class BusCacheTypeOf<Type element> : Type<
  And<[
    CPred<"$_self.isa<phy::physical::BusCacheType>()">,
    SubstLeaves<
      "$_self",
      "$_self.cast<phy::physical::BusCacheType>().getDatatype()",
      element.predicate>
  ]>,
  "a cache of " # element.summary
>;

def AnyBusCacheType : Type<
  Or<[
    BusCacheTypeOf<AnyType>.predicate,
  ]>,
  "a cache"
>;

#endif  // MLIR_PHY_DIALECT_PHYSICAL_BUS_CACHE_TYPES_TD
