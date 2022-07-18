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
include "mlir/Interfaces/InferTypeOpInterface.td"
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
    %pe = phy.pe @my_adder(%net, %buf) : (!phy.net<i32>, memref<1024xi32>) -> !phy.pe
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee,
                   Variadic<AnyCommType>:$operands);
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
// BufferOp
//===----------------------------------------------------------------------===//

def BufferOp : Phy_Op<"buf"> {
  let summary = "buffer creation operation";
  let description = [{
    The `phy.buf` operation represents a creation of a buffer that has the type
    argument as its datatype.  A buffer is a memory space that stores data.
    A buffer can be randomly accessed through a bus, or sequentially through a
    network.  It can have a device-specific attribute of location.

    Example:

    ```mlir
    %buffer = phy.buf() : memref<1024xi32>
    ```
  }];

  let results = (outs AnyStaticShapeMemRef:$buffer);

  let assemblyFormat = [{
    `(` `)` attr-dict `:` type($buffer)
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
    %net = phy.net() : !phy.net<i32>
    ```
  }];

  let results = (outs NetworkTypeOf<AnyType>:$network);

  let assemblyFormat = [{
    `(` `)` attr-dict `:` type($network)
  }];
}

//===----------------------------------------------------------------------===//
// BusOp
//===----------------------------------------------------------------------===//

def BusOp : Phy_Op<"bus"> {
  let summary = "bus creation operation";
  let description = [{
    The `phy.bus` operation represents a creation of a bus that has the type
    argument as its datatype for data transfer.  When a PE wants to access a
    buffer, there must exist a bus connecting the PE and the buffer.

    Example:

    ```mlir
    %buffer = phy.buf() : memref<1024xi32>
    %pe     = phy.pe @func(%buffer) : (memref<1024xi32>) -> !phy.pe
    %bus    = phy.bus(%buffer, %pe) : (memref<1024xi32>, !phy.pe) -> !phy.bus<i32>
    ```
  }];

  let arguments = (ins Variadic<AnyBusEndPoint>:$endpoints);
  let results = (outs BusTypeOf<AnyType>:$bus);

  let hasVerifier = 1;

  let assemblyFormat = [{
    `(` $endpoints `)` attr-dict `:` functional-type($endpoints, $bus)
  }];
}

//===----------------------------------------------------------------------===//
// #### Network APIs ####
//===----------------------------------------------------------------------===//
// ValidOp
//===----------------------------------------------------------------------===//

def ValidOp : Phy_Op<"valid"> {
  let summary = "non-blocking network receive test operation";
  let description = [{
    The `phy.valid` operation tests its operand network if the given channel
    has data to received.  This operation is non-blocking and returns a boolean
    result, which is true if there is data to receive, or false if there is not.

    Example:

    ```mlir
    %0 = phy.valid(%net : !phy.net<i32>, 1)
    ```
  }];

  let arguments = (ins NetworkTypeOf<AnyType>:$network, I32Attr:$channel);
  let results = (outs I1:$result);

  let assemblyFormat = [{
    `(` $network `:` type($network) `,` $channel `)` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// ReadyOp
//===----------------------------------------------------------------------===//

def ReadyOp : Phy_Op<"ready"> {
  let summary = "non-blocking network send test operation";
  let description = [{
    The `phy.ready` operation tests its operand network if the given channel
    is ready for sending data.  This operation is non-blocking and returns a
    boolean result, which is true if it is ready to send, or false if it is not.

    Example:

    ```mlir
    %0 = phy.ready(%net : !phy.net<i32>, 1)
    ```
  }];

  let arguments = (ins NetworkTypeOf<AnyType>:$network, I32Attr:$channel);
  let results = (outs I1:$result);

  let assemblyFormat = [{
    `(` $network `:` type($network) `,` $channel `)` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// PopOp
//===----------------------------------------------------------------------===//

def PopOp : Phy_Op<"pop", [InferTypeOpInterface]> {
  let summary = "blocking network data receive";
  let description = [{
    The `phy.pop` operation receive data from its operand network on the given
    channel.  This operation is blocking and returns the received data as its
    result.

    Example:

    ```mlir
    %0 = phy.pop(%net : !phy.net<i32>, 1)
    ```
  }];

  let arguments = (ins NetworkTypeOf<AnyType>:$network, I32Attr:$channel);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    `(` $network `:` type($network) `,` $channel `)` attr-dict
  }];

  let extraClassDeclaration = [{
    static mlir::LogicalResult inferReturnTypes(mlir::MLIRContext *context,
          llvm::Optional<::mlir::Location> location, mlir::ValueRange operands,
          mlir::DictionaryAttr attributes, mlir::RegionRange regions,
          llvm::SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
      inferredReturnTypes.assign({
          operands[0].getType().cast<NetworkType>().getDatatype()
      });
      return mlir::success();
    }
  }];
}

//===----------------------------------------------------------------------===//
// PushOp
//===----------------------------------------------------------------------===//

def PushOp : Phy_Op<"push"> {
  let summary = "blocking network data send";
  let description = [{
    The `phy.pop` operation send data to its operand network on the given
    channel.  This operation is blocking and will continue when the data is
    on the network.  It may return before the data is received from the network.
    If multiple receivers are on the channel, they will each get a copy.

    Example:

    ```mlir
    phy.push(%0 : i32, %net : !phy.net<i32>, 1)
    ```
  }];

  let arguments = (ins AnyType:$data, NetworkTypeOf<AnyType>:$network,
    I32Attr:$channel);

  let hasVerifier = 1;
  
  let assemblyFormat = [{
    `(` $data `:` type($data) `,` $network `:` type($network) `,` $channel `)` attr-dict
  }];
}

#endif  // MLIR_DIALECT_PHY_OPS_TD
