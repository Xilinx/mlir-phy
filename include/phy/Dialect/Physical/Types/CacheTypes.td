//===- CacheTypes.td ---------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_DIALECT_PHYSICAL_CACHE_TYPES_TD
#define MLIR_PHY_DIALECT_PHYSICAL_CACHE_TYPES_TD

include "phy/Dialect/Physical/PhysicalBase.td"

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

class CacheTypeOf<Type element> : Type<
  And<[
    CPred<"$_self.isa<phy::physical::CacheType>()">,
    SubstLeaves<
      "$_self",
      "$_self.cast<phy::physical::CacheType>().getDatatype()",
      element.predicate>
  ]>,
  "a cache of " # element.summary
>;

def AnyCacheType : Type<
  Or<[
    CacheTypeOf<AnyType>.predicate,
  ]>,
  "a cache"
>;

#endif  // MLIR_PHY_DIALECT_PHYSICAL_CACHE_TYPES_TD
