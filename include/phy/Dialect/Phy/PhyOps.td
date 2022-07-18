//===- PhyOps.td -------------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PHY_OPS_TD
#define MLIR_DIALECT_PHY_OPS_TD

include "phy/Dialect/Phy/Phy.td"
include "phy/Dialect/Phy/PhyTypes.td"

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/SymbolInterfaces.td"

class Phy_Op<string mnemonic, list<Trait> traits = []> :
    Op<Phy_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// PeOp
//===----------------------------------------------------------------------===//

def PeOp : Phy_Op<"pe",
    [CallOpInterface, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "processing element creation operation";
  let description = [{
    The `phy.pe` operation represents a creation of a processing element that
    has the function argument as its entry point.  The processing element will
    be free-running and the function will be invoked.  The function must be
    within the same symbol scope as the operation.  The operands must match the
    specified function type. The function is encoded as a symbol reference
    attribute named `callee`.

    Example:

    ```mlir
    %pe = phy.pe @my_adder(%net, %bus)
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee,
                   Variadic<NetworkTypeOf<AnyType>>:$operands);
  let results = (outs PeType:$pe);

  let builders = [
    OpBuilder<(ins "::mlir::func::FuncOp":$callee,
      CArg<"::mlir::ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", ::mlir::SymbolRefAttr::get(callee));
    }]>,
    OpBuilder<(ins "::mlir::SymbolRefAttr":$callee,
      CArg<"::mlir::ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", callee);
    }]>,
    OpBuilder<(ins "::mlir::StringAttr":$callee,
      CArg<"::mlir::ValueRange", "{}">:$operands), [{
      build($_builder, $_state, ::mlir::SymbolRefAttr::get(callee), operands);
    }]>];

  let extraClassDeclaration = [{
    ::mlir::FunctionType getCalleeType();

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
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, $pe)
  }];
}

//===----------------------------------------------------------------------===//
// NetworkOp
//===----------------------------------------------------------------------===//

def NetworkOp : Phy_Op<"net"> {
  let summary = "network creation operation";
  let description = [{
    The `phy.net` operation represents a creation of a network that has the type
    argument as its datatype for data transfer.  The network can be passed to PE
    functions to connect them together for message-passing based communication.

    Example:

    ```mlir
    %net = phy.net(i32)
    ```
  }];

  let results = (outs NetworkTypeOf<AnyType>:$network);

  let assemblyFormat = [{
    `(` `)` attr-dict `:` type($network)
  }];
}

#endif  // MLIR_DIALECT_PHY_OPS_TD
