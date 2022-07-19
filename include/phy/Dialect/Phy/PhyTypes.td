//===- PhyTypes.td -----------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PHY_TYPES_TD
#define MLIR_DIALECT_PHY_TYPES_TD

include "phy/Dialect/Phy/Phy.td"

include "mlir/IR/AttrTypeBase.td"

class Phy_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Phy_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

class Phy_PlaceType<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Phy_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

class Phy_RouteType<string name, string typeMnemonic, list<Trait> traits = []>
    : Phy_Type<name, typeMnemonic, traits> {
  let mnemonic = typeMnemonic;
}

class Phy_CommType<string name, string typeMnemonic, list<Trait> traits = []>
    : Phy_RouteType<name, typeMnemonic, traits> {
  let mnemonic = typeMnemonic;
}

class Phy_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Phy_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

//===----------------------------------------------------------------------===//
// PeType
//===----------------------------------------------------------------------===//

def PeType : Phy_PlaceType<"Pe", "pe"> {
  let summary = "A type specifiying a processing element";
  let description = [{
    A PE is a logical function that connects to networks or buses.  It can
    communicate with another PE or buffer via networks, or buffer via buses.
    It can have a device-specific attribute of location, e.g. a core in CPU,
    a tile or a DMA region in AIE, or a reconfiguration slot in FPGA.
  }];
}

//===----------------------------------------------------------------------===//
// NetworkType
//===----------------------------------------------------------------------===//

def NetworkType : Phy_CommType<"Network", "net"> {
  let summary = "A type specifiying a network to be connected by PEs";
  let description = [{
    A network connects multiple PEs and/or buffers for message-passing.
    It behaves as a first-in-first-out interconnect.  It can have a
    device-specific attribute of its route.  A network can be connected with
    another network using routers that provides in-transit buffers.
  }];

  let parameters = (ins "Type":$datatype);
  let assemblyFormat = "`<` $datatype `>`";
}

//===----------------------------------------------------------------------===//
// BusType
//===----------------------------------------------------------------------===//

def BusType : Phy_CommType<"Bus", "bus"> {
  let summary = "A type specifiying a bus connecting PEs to buffers";
  let description = [{
    A bus connects PEs to buffers.  A buffer has its memory address space mapped
    to the bus for PEs' access.  It can have a device-specific attribute of the
    route.  A bus can be connected to an upstream bus using caches that provides
    caching.
  }];

  let parameters = (ins "Type":$datatype);
  let assemblyFormat = "`<` $datatype `>`";
}

//===----------------------------------------------------------------------===//
// AddressedBusType
//===----------------------------------------------------------------------===//

def AddressedBusType : Phy_CommType<"AddressedBus", "addressedBus"> {
  let summary = "A type specifiying a bus with address mapping specification";
  let description = [{
    A bus with its address space used by buffers.  In the defining operation, a
    buffer can specify how its memory address space is mapped to the bus.  An
    addressed bus can be used in the same way as a 'memref' in a PE with the
    'phy.startLoad' and 'phy.startStore' operations.
  }];

  let parameters = (ins "Type":$datatype);
  let assemblyFormat = "`<` $datatype `>`";
}

//===----------------------------------------------------------------------===//
// HandleType
//===----------------------------------------------------------------------===//

def HandleType : Phy_Type<"Handle", "handle"> {
  let summary = "A type specifiying a on-going bus access";
  let description = [{
    A non-blocking bus access returns a handle, which may be waited to get the
    access result.
  }];

  let parameters = (ins "Type":$datatype);
  let assemblyFormat = "`<` $datatype `>`";
}

//===----------------------------------------------------------------------===//
// RouterType
//===----------------------------------------------------------------------===//

def RouterType : Phy_Type<"Router", "router"> {
  let summary = "A type specifiying a router connecting two networks";
  let description = [{
    A network can be connected with another network using routers that provides
    in-transit buffers.  The parameter 'depth' specifies the depth of the fifo
    buffer in the router.
  }];

  let parameters = (ins "Type":$datatype, "int":$depth);
  let assemblyFormat = "`<` $datatype `,` $depth `>`";
}

//===----------------------------------------------------------------------===//
// CacheType
//===----------------------------------------------------------------------===//

def CacheType : Phy_Type<"Cache", "cache"> {
  let summary = "A type specifiying a cache connecting two buses";
  let description = [{
    A bus can be connected to an upstream bus using caches that provides caching.
    The parameter 'size' specifies the elements to be stored in the cache.
  }];

  let parameters = (ins "Type":$datatype, "int":$size);
  let assemblyFormat = "`<` $datatype `,` $size `>`";
}

//===----------------------------------------------------------------------===//
// Type Constraints
//===----------------------------------------------------------------------===//

class NetworkTypeOf<Type element> : Type<
  And<[
    CPred<"$_self.isa<NetworkType>()">,
    SubstLeaves<
      "$_self",
      "$_self.cast<NetworkType>().getDatatype()",
      element.predicate>
  ]>,
  "a network of " # element.summary
>;

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

class AddressedBusTypeOf<Type element> : Type<
  And<[
    CPred<"$_self.isa<AddressedBusType>()">,
    SubstLeaves<
      "$_self",
      "$_self.cast<AddressedBusType>().getDatatype()",
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

class RouterTypeOf<Type element> : Type<
  And<[
    CPred<"$_self.isa<RouterType>()">,
    SubstLeaves<
      "$_self",
      "$_self.cast<RouterType>().getDatatype()",
      element.predicate>
  ]>,
  "a router of " # element.summary
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

def AnyPeArgsType : Type<
  Or<[
    NetworkTypeOf<AnyType>.predicate,
    AnyStaticShapeMemRef.predicate,
    AddressedBusTypeOf<AnyType>.predicate,
  ]>,
  "a memref, a network or an addressed bus"
>;

def AnyLoadStoreType : Type<
  Or<[
    AnyStaticShapeMemRef.predicate,
    AddressedBusTypeOf<AnyType>.predicate,
  ]>,
  "a memref or an addressed bus"
>;

def AnyBusEndPoint : Type<
  Or<[
    AnyStaticShapeMemRef.predicate,
    PeType.predicate
  ]>,
  "a memref or a pe"
>;

def AnyPlacableType : Type<
  Or<[
    AnyStaticShapeMemRef.predicate,
    PeType.predicate,
    RouterTypeOf<AnyType>.predicate,
    CacheTypeOf<AnyType>.predicate
  ]>,
  "a memref, a network, a router or a cache"
>;

def AnyRoutableType : Type<
  Or<[
    NetworkTypeOf<AnyType>.predicate,
    BusTypeOf<AnyType>.predicate,
    AddressedBusTypeOf<AnyType>.predicate
  ]>,
  "a network, a bus, or an addressed bus"
>;

#endif  // MLIR_DIALECT_PHY_TYPES_TD
