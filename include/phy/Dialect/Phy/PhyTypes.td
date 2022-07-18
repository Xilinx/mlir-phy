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
// Type Constraints
//===----------------------------------------------------------------------===//

class NetworkTypeOf<Type element> : Type<
  And<[
    CPred<"$_self.isa<NetworkType>()">,
    SubstLeaves<
      "$_self",
      "$_self.cast<NetworkType>().datatype().value()",
      element.predicate>
  ]>,
  "network of " # element.summary
>;

def AnyCommType : Type<
  Or<[
    NetworkTypeOf<AnyType>.predicate,
    AnyStaticShapeMemRef.predicate
  ]>,
  "a memref or a network"
>;

#endif  // MLIR_DIALECT_PHY_TYPES_TD
