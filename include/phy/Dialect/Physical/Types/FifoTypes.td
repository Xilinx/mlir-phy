//===- FifoTypes.td ----------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_DIALECT_PHYSICAL_FIFO_TYPES_TD
#define MLIR_PHY_DIALECT_PHYSICAL_FIFO_TYPES_TD

include "phy/Dialect/Physical/PhysicalBase.td"

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

class FifoTypeOf<Type element> : Type<
  And<[
    CPred<"$_self.isa<phy::physical::FifoType>()">,
    SubstLeaves<
      "$_self",
      "$_self.cast<phy::physical::FifoType>().getDatatype()",
      element.predicate>
  ]>,
  "a fifo of " # element.summary
>;

def AnyFifoType : Type<
  Or<[
    FifoTypeOf<AnyType>.predicate,
  ]>,
  "a fifo"
>;

#endif  // MLIR_PHY_DIALECT_PHYSICAL_FIFO_TYPES_TD
