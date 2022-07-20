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
                   Variadic<AnyPeArgsType>:$operands);
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
    %0 = phy.valid<1>(%net : !phy.net<i32>)
    ```
  }];

  let arguments = (ins NetworkTypeOf<AnyType>:$network, I32Attr:$channel);
  let results = (outs I1:$result);

  let assemblyFormat = [{
    `<` $channel `>` `(` $network `:` type($network) `)` attr-dict
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
    %0 = phy.ready<1>(%net : !phy.net<i32>)
    ```
  }];

  let arguments = (ins NetworkTypeOf<AnyType>:$network, I32Attr:$channel);
  let results = (outs I1:$result);

  let assemblyFormat = [{
    `<` $channel `>` `(` $network `:` type($network) `)` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// PopOp
//===----------------------------------------------------------------------===//

def PopOp : Phy_Op<"pop",
    [TypesMatchWith<"'network' data has the same type as result",
                    "result", "network",
                    "NetworkType::get($_self.getContext(), $_self)">]> {

  let summary = "blocking network data receive";
  let description = [{
    The `phy.pop` operation receive data from its operand network on the given
    channel.  This operation is blocking and returns the received data as its
    result.

    Example:

    ```mlir
    %0 = phy.pop<1>(%net) : i32
    ```
  }];

  let arguments = (ins NetworkTypeOf<AnyType>:$network, I32Attr:$channel);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    `<` $channel `>` `(` $network `)` attr-dict `:` type($result) 
  }];
}

//===----------------------------------------------------------------------===//
// PushOp
//===----------------------------------------------------------------------===//

def PushOp : Phy_Op<"push",
    [TypesMatchWith<"'network' data has the same type as 'data'",
                    "data", "network",
                    "NetworkType::get($_self.getContext(), $_self)">]> {

  let summary = "blocking network data send";
  let description = [{
    The `phy.pop` operation send data to its operand network on the given
    channel.  This operation is blocking and will continue when the data is
    on the network.  It may return before the data is received from the network.
    If multiple receivers are on the channel, they will each get a copy.

    Example:

    ```mlir
    phy.push<1>(%0 : i32, %net)
    ```
  }];

  let arguments = (ins AnyType:$data, NetworkTypeOf<AnyType>:$network,
    I32Attr:$channel);
  
  let assemblyFormat = [{
    `<` $channel `>` `(` $data `:` type($data) `,` $network `)` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// #### Bus APIs ####
//===----------------------------------------------------------------------===//
// StartLoadOp
//===----------------------------------------------------------------------===//

def StartLoadOp : Phy_Op<"startLoad",
    [TypesMatchWith<"result's type is the handle of element type of 'memref'",
                    "memref", "result",
                    "HandleType::get($_self.getContext(), "
                      "($_self.isa<MemRefType>() ? ("
                        "$_self.cast<MemRefType>().getElementType()"
                      ") : ("
                        "$_self.cast<AddressedBusType>().getDatatype()"
                      ")))">,
     MemRefsNormalizable]> {

  let summary = "non-blocking load of the data";
  let description = [{
    A non-blocking bus access that reads the data from a buffer, or an addressed
    bus, as specified in 'memref'.  This operation returns a handle, which may be
    waited using `phy.wait` to get the access result.

    Example:

    ```mlir
    %load_handle = phy.startLoad %mem[%idx] : memref<1024xi32>
    %0 = phy.wait(%load_handle) : i32
    ```
  }];

  let arguments = (ins AnyLoadStoreType:$memref,
                       Variadic<Index>:$indices);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    $memref `[` $indices `]` attr-dict `:` type($memref)
  }];
}

//===----------------------------------------------------------------------===//
// StartStoreOp
//===----------------------------------------------------------------------===//

def StartStoreOp : Phy_Op<"startStore",
    [TypesMatchWith<"type of 'value' matches element type of 'memref'",
                    "memref", "value",
                    "$_self.isa<MemRefType>() ? ("
                        "$_self.cast<MemRefType>().getElementType()"
                    ") : ("
                        "$_self.cast<AddressedBusType>().getDatatype()"
                    ")">,
     TypesMatchWith<"result's type is a handle of NoneType",
                    "memref", "handle",
                    "HandleType::get($_self.getContext(), "
                      "NoneType::get($_self.getContext()))">,
     MemRefsNormalizable]> {

  let summary = "non-blocking store of the data";
  let description = [{
    A non-blocking bus access that stores the data to a buffer, or an addressed
    bus, as specified in 'memref'.  This operation returns a handle, which may
    be waited using `phy.wait`.

    Example:

    ```mlir
    %store_handle = phy.startStore %0, %mem[%idx] : memref<1024xi32>
    phy.wait(%store_handle) : none
    ```
  }];

  let arguments = (ins AnyType:$value,
                       AnyLoadStoreType:$memref,
                       Variadic<Index>:$indices);
  let results = (outs HandleTypeOf<NoneType>:$handle);

  let assemblyFormat = [{
    $value `,` $memref `[` $indices `]` attr-dict `:` type($memref)
  }];
}

//===----------------------------------------------------------------------===//
// WaitOp
//===----------------------------------------------------------------------===//

def WaitOp : Phy_Op<"wait",
    [TypesMatchWith<"handle's data type matches result's type",
                    "result", "handle",
                    "HandleType::get($_self.getContext(), $_self)">]> {

  let summary = "blocking wait until a handle is ready";
  let description = [{
    A non-blocking bus access returns a handle, which may be waited using this
    operation to get the access result.  For store access, none is returned.

    Example:

    ```mlir
    %0 = phy.wait(%load_handle) : i32
    phy.wait(%store_handle) : none
    ```
  }];

  let arguments = (ins HandleTypeOf<AnyType>:$handle);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    `(` $handle `)` attr-dict `:` type($result) 
  }];
}

//===----------------------------------------------------------------------===//
// #### Bridging APIs ####
//===----------------------------------------------------------------------===//
// RouterOp
//===----------------------------------------------------------------------===//

def RouterOp : Phy_Op<"router",
    [TypesMatchWith<"type of 'network1' matches element type of the router",
                    "router", "network1",
                    "NetworkType::get($_self.getContext(), "
                      "$_self.cast<RouterType>().getDatatype())">,
     TypesMatchWith<"type of 'network2' matches element type of the router",
                    "router", "network2",
                    "NetworkType::get($_self.getContext(), "
                      "$_self.cast<RouterType>().getDatatype())">]> {

  let summary = "creation of a router connecting two networks";
  let description = [{
    An operation creating a router to connect two networks.  With the
    connection, the data sent on either network will be received on the other.
    The 'phy.router' type specifies the fifo size of the router.

    Example:

    ```mlir
    %net1   = phy.net() : !phy.net<i32>
    %net2   = phy.net() : !phy.net<i32>
    %router = phy.router(%net1, %net2) : !phy.router<i32, 2>
    ```
  }];

  let arguments = (ins NetworkTypeOf<AnyType>:$network1,
                       NetworkTypeOf<AnyType>:$network2);
  let results = (outs RouterTypeOf<AnyType>:$router);

  let assemblyFormat = [{
    `(` $network1 `,` $network2 `)` attr-dict `:` type($router)
  }];
}

//===----------------------------------------------------------------------===//
// CacheOp
//===----------------------------------------------------------------------===//

def CacheOp : Phy_Op<"cache",
    [TypesMatchWith<"type of 'upstream' matches element type of the cache",
                    "cache", "upstream",
                    "BusType::get($_self.getContext(), "
                      "$_self.cast<CacheType>().getDatatype())">,
     TypesMatchWith<"type of 'downstream' matches element type of the cache",
                    "cache", "downstream",
                    "BusType::get($_self.getContext(), "
                      "$_self.cast<CacheType>().getDatatype())">]> {

  let summary = "creation of a cache layer between two buses";
  let description = [{
    An operation creating a cache to connect two buses.  With the cache,
    the memory access on the 'downstream' bus from the buffers on the 'upstream'
    bus will be cached, and the 'downstream' bus is able to receive data from
    the buffers on the 'upstream' bus.  The 'phy.cache' type specifies the
    number of elements that can be cached.

    Example:

    ```mlir
    %bus1   = phy.bus(%buffer) : (memref<1024xi32>) -> !phy.bus<i32>
    %bus2   = phy.bus(%pe) : (!phy.pe) -> !phy.bus<i32>
    %cache  = phy.cache(%bus1, %bus2) : !phy.cache<i32, 1024>
    ```
  }];

  let arguments = (ins BusTypeOf<AnyType>:$upstream,
                       BusTypeOf<AnyType>:$downstream);
  let results = (outs CacheTypeOf<AnyType>:$cache);

  let assemblyFormat = [{
    `(` $upstream `,` $downstream `)` attr-dict `:` type($cache)
  }];
}

//===----------------------------------------------------------------------===//
// AddressedCacheOp
//===----------------------------------------------------------------------===//

def AddressedCacheOp : Phy_Op<"addressedCache",
    [TypesMatchWith<"type of 'upstream' matches element type of the cache",
                    "cache", "upstream",
                    "AddressedBusType::get($_self.getContext(), "
                      "$_self.cast<CacheType>().getDatatype())">,
     TypesMatchWith<"type of 'downstream' matches element type of the cache",
                    "cache", "downstream",
                    "AddressedBusType::get($_self.getContext(), "
                      "$_self.cast<CacheType>().getDatatype())">]> {

  let summary = "creation of a cache layer between two addressed buses";
  let description = [{
    An operation creating a cache to connect two buses.  With the cache,
    the memory access on the 'downstream' bus from the buffers on the 'upstream'
    bus will be cached, and the 'downstream' bus is able to receive data from
    the buffers on the 'upstream' bus.  The 'phy.cache' type specifies the
    number of elements that can be cached.

    Example:

    ```mlir
    %abus1 = phy.addressedBus() : !phy.addressedBus<i32>
    %abus2 = phy.addressedBus() : !phy.addressedBus<i32>
    %cache  = phy.addressedCache(%abus1, %abus2) : !phy.cache<i32, 1024>
    ```
  }];

  let arguments = (ins AddressedBusTypeOf<AnyType>:$upstream,
                       AddressedBusTypeOf<AnyType>:$downstream);
  let results = (outs CacheTypeOf<AnyType>:$cache);

  let assemblyFormat = [{
    `(` $upstream `,` $downstream `)` attr-dict `:` type($cache)
  }];
}

//===----------------------------------------------------------------------===//
// #### Placement and Routing APIs ####
//===----------------------------------------------------------------------===//
// EndOp
//===----------------------------------------------------------------------===//

def EndOp : Phy_Op<"end", [Terminator, ParentOneOf<["PlatformOp", "DeviceOp"]>]> {
  let summary = "end of usage of platform or device";
  let description = [{
    'phy.end' is an implied terminator for the 'phy.platform' or 'phy.device'
    regions.  You do not need to manually write it.
  }];
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// PlatformOp
//===----------------------------------------------------------------------===//

def PlatformOp : Phy_Op<"platform",
    [SingleBlockImplicitTerminator<"phy::EndOp">,
     NoRegionArguments]> {

  let summary = "usage of a platform";
  let description = [{
    An operation that uses an abstract physical platform for the enclosing region.
    The region is used to place and route elements onto the platform.  The
    locations for placement, and the wires for routing, are dependent on which
    platform is used.

    Example:

    ```mlir
    phy.platform<"xilinx"> {
      phy.device<"hls"> {}
      phy.route<"pl-aie">(%net1)
    }
    ```
  }];

  let arguments = (ins StrAttr:$platform);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{ `<` $platform `>` regions attr-dict }];
}

//===----------------------------------------------------------------------===//
// DeviceOp
//===----------------------------------------------------------------------===//

def DeviceOp : Phy_Op<"device",
    [SingleBlockImplicitTerminator<"phy::EndOp">,
     NoRegionArguments, HasParent<"PlatformOp">]> {

  let summary = "usage of a device";
  let description = [{
    An operation that uses an abstract physical device for the enclosing region.
    The region is used to place and route elements onto the device.  The
    locations for placement, and the wires for routing, are dependent on which
    device is used.  This operation must be enclosed by a 'phy.platform' operation.

    Example:

    ```mlir
    phy.platform<"xilinx"> {
      phy.device<"hls"> {
        phy.place<"slr0">(%pe)
      }
    }
    ```
  }];

  let arguments = (ins StrAttr:$device);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{ `<` $device `>` regions attr-dict }];
}

//===----------------------------------------------------------------------===//
// PlaceOp
//===----------------------------------------------------------------------===//

def PlaceOp : Phy_Op<"place", [HasParent<"DeviceOp">]> {
  let summary = "placement of an element";
  let description = [{
    An operation that place the 'element' to the specific location of the
    enclosing device.  This operation must be enclosed by a 'phy.device'
    operation.  The 'location' argument is device-dependent.  Please check
    the documentation of the platform and device plugins.

    Example:

    ```mlir
    phy.platform<"xilinx"> {
      phy.device<"hls"> {
        phy.place<"slr0">(%pe)
      }
    }
    ```
  }];

  let arguments = (ins StrAttr:$location, AnyPlacableType:$element);
  let assemblyFormat = [{
    `<` $location `>` `(` $element `:` type($element)  `)` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// RouteOp
//===----------------------------------------------------------------------===//

def RouteOp : Phy_Op<"route"> {
  let summary = "route of a communication path";
  let description = [{
    An operation that route the 'communication' to the specific wires of the
    enclosing device, platform, or global connections.  The 'wires' argument
    is device-dependent, platform-dependent, or global-dependent.  Please check
    the documentation of the platform and device plugins, or the global plugin.

    Example:

    ```mlir
    phy.platform<"xilinx"> {
      phy.device<"hls"> {
        phy.place<"slr0">(%pe)
      }
    }
    ```
  }];

  let arguments = (ins StrArrayAttr:$wires, AnyRoutableType:$communication);
  let assemblyFormat = [{
    `<` $wires `>` `(` $communication `:` type($communication)  `)` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// #### Address Lowering APIs ####
//===----------------------------------------------------------------------===//
// AddressedBusOp
//===----------------------------------------------------------------------===//

def AddressedBusOp : Phy_Op<"addressedBus"> {
  let summary = "addressed bus creation operation";
  let description = [{
    The `phy.addressedBus` operation represents a creation of an addressed bus
    that can have buffers mapped to its memory space using the 'phy.mmap'
    operation.

    Example:

    ```mlir
    %buf = phy.buf() : memref<1024xi32>
    %abus = phy.addressedBus() : !phy.addressedBus<i32>
    phy.mmap(%abus[10:15], %buf[20:] : memref<1024xi32>)
    %pe = phy.pe @func(%abus) : (!phy.addressedBus<i32>) -> !phy.pe
    ```
  }];

  let arguments = (ins);
  let results = (outs AddressedBusTypeOf<AnyType>:$bus);

  let assemblyFormat = [{
    `(` `)` attr-dict `:` type($bus)
  }];
}

//===----------------------------------------------------------------------===//
// MmapOp
//===----------------------------------------------------------------------===//

def MmapOp : Phy_Op<"mmap",
    [TypesMatchWith<"type of 'bus' matches element type of the 'buffer'",
                    "buffer", "bus",
                    "AddressedBusType::get($_self.getContext(), "
                      "$_self.cast<MemRefType>().getElementType())">]> {

  let summary = "creation of a buffer into addressed bus memory space";
  let description = [{
    An operation that maps the 'buffer' starting the 'offset'-th element,
    into the 'bus'.  The mapped address is from 'begin'-th element
    (inclusive), to the 'end'-th element (exclusive) on the bus.

    Example:

    ```mlir
    phy.mmap(%abus[10:15], %buf[20:] : memref<1024xi32>)
    // abus[10] will be buf[20], abus[11] will be buf[21], ...
    ```
  }];

  let arguments = (ins AddressedBusTypeOf<AnyType>:$bus,
                       I32Attr:$begin,
                       I32Attr:$end,
                       AnyStaticShapeMemRef:$buffer,
                       I32Attr:$offset);

  let assemblyFormat = [{
    `(` $bus `[` $begin `:` $end `]` `,` $buffer `[` $offset `:` `]`
    `:` type($buffer) `)` attr-dict 
  }];
}

#endif  // MLIR_DIALECT_PHY_OPS_TD
