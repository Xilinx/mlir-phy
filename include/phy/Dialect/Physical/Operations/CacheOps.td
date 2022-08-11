//===- CacheOps.td -----------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_DIALECT_PHYSICAL_CACHE_OPS_TD
#define MLIR_PHY_DIALECT_PHYSICAL_CACHE_OPS_TD

include "phy/Dialect/Physical/PhysicalBase.td"

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

#endif  // MLIR_PHY_DIALECT_PHYSICAL_CACHE_OPS_TD
