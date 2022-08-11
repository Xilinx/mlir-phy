//===- BasicOps.td -----------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_DIALECT_PHYSICAL_BASIC_OPS_TD
#define MLIR_PHY_DIALECT_PHYSICAL_BASIC_OPS_TD

include "phy/Dialect/Physical/PhysicalBase.td"

include "mlir/Interfaces/CallInterfaces.td"

//===----------------------------------------------------------------------===//
// BufferOp
//===----------------------------------------------------------------------===//

def BufferOp : Physical_Op<"buffer"> {
  let summary = "buffer creation operation";
  let description = [{
    The `physical.buffer` operation represents a creation of a buffer that has
    the type argument as its datatype.  A buffer is a memory space that stores
    data.  A buffer can be randomly accessed.  It can have a device-specific
    attribute of location.

    Example:

    ```mlir
    %buffer = physical.buffer() : memref<1024xi32>
    ```
  }];

  let results = (outs AnyStaticShapeMemRef:$buffer);

  let assemblyFormat = [{
    `(` `)` attr-dict `:` type($buffer)
  }];
}

//===----------------------------------------------------------------------===//
// CoreOp
//===----------------------------------------------------------------------===//

def CoreOp : Physical_Op<"core",
    [CallOpInterface, DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {

  let summary = "processing core creation operation";
  let description = [{
    The `physical.core` operation represents a creation of a processing core
    that has the function argument as its entry point.  The processing core
    will be free-running and the function will be invoked.  The function must
    be within the same symbol scope as the operation.  The operands must match
    the specified function type. The function is encoded as a symbol reference
    attribute named `callee`.

    Example:

    ```mlir
    %core = physical.core @my_adder(%buf)
          : (memref<1024xi32>) -> !physical.core
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee,
                   Variadic<AnyCoreArgsType>:$operands);
  let results = (outs CoreType:$core);

  let extraClassDeclaration = [{
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
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, $core)
  }];
}

#endif  // MLIR_PHY_DIALECT_PHYSICAL_BASIC_OPS_TD
