<!-- Autogenerated by mlir-tblgen; don't manually edit -->
# 'physical' Dialect

This dialect describes the actual physical implementation of the spatial
architecture. It abstracted out the most common low-level features provided
by the devices and platforms.  The layout information is embedded into the
node as device-specific attributes.

The physical operations used in a valid code must be non-overlapping with
others and has a specific physical location.  For sharing of a resource,
a shared version is implemented depending on device support, and only one
shared operation is specified in the code.

The dialect acts as the assembly for physical designs.  The mapping from
this dialect to either code generator (mlir-translate), or the lower level
dialects must be one-to-one.  Depending on the device and platform, some
operations defined might not be supported.

[TOC]

## Operation definition

### `physical.buffer` (::phy::physical::BufferOp)

buffer creation operation


Syntax:

```
operation ::= `physical.buffer` `(` `)` attr-dict `:` type($buffer)
```

The `physical.buffer` operation represents a creation of a buffer that has
the type argument as its datatype.  A buffer is a memory space that stores
data.  A buffer can be randomly accessed.  It can have a device-specific
attribute of location.

Example:

```mlir
%buffer = physical.buffer() : memref<1024xi32>
```

#### Results:

| Result | Description |
| :----: | ----------- |
| `buffer` | statically shaped memref of any type values

### `physical.bus_cache` (::phy::physical::BusCacheOp)

creation of a cache layer between two addressed buses


Syntax:

```
operation ::= `physical.bus_cache` `(` $upstream `,` $downstream `)` attr-dict `:` type($bus_cache)
```

An operation creating a cache to connect two buses.  With the bus cache,
the memory access on the 'downstream' bus from the buffers on the 'upstream'
bus will be cached, and the 'downstream' bus is able to receive data from
the buffers on the 'upstream' bus.  The 'physical.bus_cache' type specifies
the number of elements that can be cached.

Example:

```mlir
%bus1 = physical.bus() : !physical.bus<i32>
%bus2 = physical.bus() : !physical.bus<i32>
%cache = physical.bus_cache(%bus1, %bus2) : !physical.bus_cache<i32, 1024>
```

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `upstream` | a bus of any type
| `downstream` | a bus of any type

#### Results:

| Result | Description |
| :----: | ----------- |
| `bus_cache` | a cache of any type

### `physical.bus_mmap` (::phy::physical::BusMmapOp)

creation of a buffer into an addressed bus memory space


Syntax:

```
operation ::= `physical.bus_mmap` `(` $bus `[` $begin `:` $end `]` `,` $buffer `[` $offset `:` `]`
              `:` type($buffer) `)` attr-dict
```

An operation that maps the 'buffer' starting the 'offset'-th element,
into the 'bus'.  The mapped address is from 'begin'-th element
(inclusive), to the 'end'-th element (exclusive) on the bus.

Example:

```mlir
physical.bus_mmap(%bus[10:15], %buf[20:] : memref<1024xi32>)
// bus[10] will be buf[20], bus[11] will be buf[21], ...
```

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `begin` | ::mlir::IntegerAttr | 64-bit signless integer attribute
| `end` | ::mlir::IntegerAttr | 64-bit signless integer attribute
| `offset` | ::mlir::IntegerAttr | 64-bit signless integer attribute

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `bus` | a bus of any type
| `buffer` | statically shaped memref of any type values

### `physical.bus` (::phy::physical::BusOp)

addressed bus creation operation


Syntax:

```
operation ::= `physical.bus` `(` `)` attr-dict `:` type($bus)
```

The `physical.bus` operation represents a creation of an addressed bus
that can have buffers mapped to its memory space using the
'physical.bus_mmap' operation.

Example:

```mlir
%buf = physical.buffer() : memref<1024xi32>
%bus = physical.bus() : !physical.bus<i32>
physical.bus_mmap(%bus[10:15], %buf[20:] : memref<1024xi32>)
%pe = physical.core @func(%bus) : (!physical.bus<i32>) -> !physical.core
```

#### Results:

| Result | Description |
| :----: | ----------- |
| `bus` | a bus of any type

### `physical.core` (::phy::physical::CoreOp)

processing core creation operation


Syntax:

```
operation ::= `physical.core` $callee `(` $operands `)` attr-dict `:` functional-type($operands, $core)
```

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

Interfaces: CallOpInterface, SymbolUserOpInterface

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `callee` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `operands` | a memref, a bus, a stream endpoint or a lock

#### Results:

| Result | Description |
| :----: | ----------- |
| `core` | A type specifiying a processing core

### `physical.end` (::phy::physical::EndOp)

end of usage of a region


Syntax:

```
operation ::= `physical.end` attr-dict
```

'physical.end' is an implied terminator.

Traits: Terminator

### `physical.lock_acquire` (::phy::physical::LockAcquireOp)

lock acquisition operation


Syntax:

```
operation ::= `physical.lock_acquire` `<` $state `>`  `(` $lock `)` attr-dict
```

The `physical.lock_acquire` operation acquires a lock when the lock is
released in the specified `state`.  It is blocking and returns only when
the lock is acquired, and other users cannot acquire the lock until it
is released.

Example:
```
  physical.lock_acquire<0>(%lock)
```

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `state` | ::mlir::IntegerAttr | 64-bit signless integer attribute

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `lock` | An atomic synchronization element

### `physical.lock` (::phy::physical::LockOp)

lock creation operation


Syntax:

```
operation ::= `physical.lock` `<` $state `>`  `(` `)` attr-dict
```

The `physical.lock` operation represents a creation of a lock.  A lock is
an atomic unit that can be used to limit access to a resource.  This
operation returns the created lock.  `state` specifies the initially
released state of the lock when the system is up.

Example:
```
  %lock = physical.lock<0>()
```

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `state` | ::mlir::IntegerAttr | 64-bit signless integer attribute

#### Results:

| Result | Description |
| :----: | ----------- |
| `lock` | An atomic synchronization element

### `physical.lock_release` (::phy::physical::LockReleaseOp)

lock releasing operation


Syntax:

```
operation ::= `physical.lock_release` `<` $state `>`  `(` $lock `)` attr-dict
```

The `physical.lock_release` operation release a lock to the specified
`state`.  Once the lock is released, it can be acquired by other users.

Example:
```
  physical.lock_release<0>(%lock)
```

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `state` | ::mlir::IntegerAttr | 64-bit signless integer attribute

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `lock` | An atomic synchronization element

### `physical.start_load` (::phy::physical::StartLoadOp)

non-blocking load of the data


Syntax:

```
operation ::= `physical.start_load` $memref `[` $indices `]` attr-dict `:` type($memref)
```

A non-blocking bus access that reads the data from a buffer, or an addressed
bus, as specified in 'memref'.  This operation returns a handle, which may be
waited using `physical.wait` to get the access result.

Example:

```mlir
%load_handle = physical.start_load %mem[%idx] : memref<1024xi32>
%0 = physical.wait(%load_handle) : i32
```

Traits: MemRefsNormalizable

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `memref` | a memref or a bus
| `indices` | index

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | any type

### `physical.start_store` (::phy::physical::StartStoreOp)

non-blocking store of the data


Syntax:

```
operation ::= `physical.start_store` $value `,` $memref `[` $indices `]` attr-dict `:` type($memref)
```

A non-blocking bus access that stores the data to a buffer, or an addressed
bus, as specified in 'memref'.  This operation returns a handle, which may
be waited using `physical.wait`.

Example:

```mlir
%store_handle = physical.start_store %0, %mem[%idx] : memref<1024xi32>
physical.wait(%store_handle) : none
```

Traits: MemRefsNormalizable

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `value` | any type
| `memref` | a memref or a bus
| `indices` | index

#### Results:

| Result | Description |
| :----: | ----------- |
| `handle` | an async handle of none type

### `physical.stream_dma_connect` (::phy::physical::StreamDmaConnectOp)

creation of a dma connection that connects to a buffer


Syntax:

```
operation ::= `physical.stream_dma_connect` (`<` $tag^ `>`)?
              `(` $lock `[` $acquire `->` $release `]` `,`
              $buffer `[` $start `:` $end `]` `:` type($buffer)
              ( `,` $next^ )?
              `)` regions attr-dict
```

`physical.stream_dma_connect` connects a buffer to/from the stream endpoint
and when the data transfer in this connection is completed, the next
connection as specified in `next` will be established.  If the `next` is
not specified, the DMA engine will be terminated when the current
connection is completed.

In each connection, the `buffer` is specified with the `start` point,
inclusively, and the `end` point, exclusively.  The data from the
buffer/stream will be transferred to the stream/buffer, and the lock will
be acquired from the `acquire` state before the transfer is performed, and
released to the `release` state when the transfer is done.

A connection can have an optional `tag` attribute.  When the `tag` is
specified, an output connection will have the stream data tagged with
`tag`.  For input connections, the `tag` will be ignored and all data
received from the stream will be part of the connection.

Example:

```mlir
%0 = physical.stream_dma_connect<1>(
  %lock[0->1], %buffer[0:1024]: memref<1024xi32>, %0)
}
```

Traits: HasParent<StreamDmaOp>

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `tag` | ::mlir::IntegerAttr | 64-bit signless integer attribute
| `acquire` | ::mlir::IntegerAttr | 64-bit signless integer attribute
| `release` | ::mlir::IntegerAttr | 64-bit signless integer attribute
| `start` | ::mlir::IntegerAttr | 64-bit signless integer attribute
| `end` | ::mlir::IntegerAttr | 64-bit signless integer attribute

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `lock` | An atomic synchronization element
| `buffer` | statically shaped memref of any type values
| `next` | A stream DMA connection

#### Results:

| Result | Description |
| :----: | ----------- |
| `connection` | A stream DMA connection

### `physical.stream_dma` (::phy::physical::StreamDmaOp)

creation of a dma engine that connects buffers with a stream


Syntax:

```
operation ::= `physical.stream_dma` `(` $endpoint `:` type($endpoint) `)` regions attr-dict
```

An operation creating a stream dma engines that is connected to a stream
endpoint.  If the endpoint is an istream, then the stream's data is written
to the buffers sequentially according to the order as specified in the
`physical.stream_dma_connect` operation.  Otherwise, if the endpoint is an
ostream, the buffers are read from instead.  Depending on the target, one
or multiple `physical.stream_dma_connect` operations are supported in a
single `physical.stream_dma` region.  The first connection in the region is
established first, and the next connection as specified in the operation is
established next, so on and so forth.

Example:

```mlir
physical.stream_dma(%endpoint: !physical.istream<i32>) {
  %0 = physical.stream_dma_connect(
    %lock1[0->1], %buffer1[0:1024]: memref<1024xi32>, %1)
  %1 = physical.stream_dma_connect(
    %lock2[0->1], %buffer2[0:1024]: memref<1024xi32>, %0)
}
```

Traits: SingleBlockImplicitTerminator<EndOp>

Interfaces: RegionKindInterface

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `endpoint` | a stream endpoint

### `physical.stream_hub` (::phy::physical::StreamHubOp)

creation of a stream hub connecting multiple streams


Syntax:

```
operation ::= `physical.stream_hub` `(` $endpoints `)` attr-dict `:` functional-type($endpoints, $stream_hub)
```

An operation creating a stream hub to connect multiple streams.  A stream
reads data from all `phy.istream` endpoints, and broadcast data to all
`phy.ostream` endpoints.  Depending on the target, one or multiple input
or output endpoints are supported.  Depending on the target, multicasting
using data tags may be supported and only streams observing the tag of a
piece of data will receive the data.

Example:

```mlir
%stream_hub = physical.stream_hub(%src1, %src2, %dest1, %dest2)
        : (!physical.istream<i32>,  !physical.istream<i32>,
           !physical.ostream<i32>, !physical.ostream<i32>)
        -> !physical.stream_hub<i32>
```

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `endpoints` | a stream endpoint

#### Results:

| Result | Description |
| :----: | ----------- |
| `stream_hub` | a stream hub

### `physical.stream` (::phy::physical::StreamOp)

streaming connection creation operation


Syntax:

```
operation ::= `physical.stream` (`<` $tags^ `>`)? `(` `)` attr-dict
              `:` `(` type($ostream) `,` type($istream) `)`
```

The `physical.stream` operation represents a creation of a stream that
connects two endpoints and provides a streaming connection.  The created
stream can be connected as an operand in `physical.core` operations for the
software function to access and communicate, or by a `physical.stream_dma`
operation.  Streams can be connected using `physical.stream_hub`s.  A
stream can optionally observe tagged data, and the observed tags shall be
specified as the `tags` attribute.  Data's tag will be preseved when
passing from the ostream endpoint to the istream endpoint.

Example:

```mlir
%stream:2 = physical.stream<[0,1]>()
          : (!physical.ostream<i32>, !physical.istream<i32>)
// %stream#0 is the ostream, and %stream#1 is the istream
```

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `tags` | ::mlir::ArrayAttr | 64-bit integer array attribute

#### Results:

| Result | Description |
| :----: | ----------- |
| `ostream` | a writing endpoint (ostream) of a stream
| `istream` | a reading endpoint (istream) of a stream

### `physical.wait` (::phy::physical::WaitOp)

blocking wait until a handle is ready


Syntax:

```
operation ::= `physical.wait` `(` $handle `)` attr-dict `:` type($result)
```

A non-blocking bus access returns a handle, which may be waited using this
operation to get the access result.  For store access, none is returned.

Example:

```mlir
%0 = physical.wait(%load_handle) : i32
physical.wait(%store_handle) : none
```

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `handle` | a handle

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | any type

## Type definition

### AsyncHandleType

A type specifiying a on-going memory access

Syntax:

```
!physical.async_handle<
  Type   # datatype
>
```

A non-blocking memory access returns a handle, which may be waited to get
the access result.

#### Parameters:

| Parameter | C++ type | Description |
| :-------: | :-------: | ----------- |
| datatype | `Type` |  |

### BusCacheType

A type specifiying a cache connecting two buses

Syntax:

```
!physical.bus_cache<
  Type,   # datatype
  int   # size
>
```

A bus can be connected to an upstream bus using caches that provides caching.
The parameter 'size' specifies the elements to be stored in the cache.

#### Parameters:

| Parameter | C++ type | Description |
| :-------: | :-------: | ----------- |
| datatype | `Type` |  |
| size | `int` |  |

### BusType

A type specifiying a bus with address mapping specification

Syntax:

```
!physical.bus<
  Type   # datatype
>
```

A bus with its address space used by buffers.  In the defining operation, a
buffer can specify how its memory address space is mapped to the bus.  A
bus can be used in the same way as a 'memref' in a PE with the
'physical.start_load' and 'physical.start_store' operations.

#### Parameters:

| Parameter | C++ type | Description |
| :-------: | :-------: | ----------- |
| datatype | `Type` |  |

### CoreType

A type specifiying a processing core

Syntax: `!physical.core`

A core is a logical function that computes.  It can have a device-specific
attribute of location, e.g. a core in CPU or a tile core in AIE.

### IStreamType

A type specifiying a streaming endpoint for input

Syntax:

```
!physical.istream<
  Type   # datatype
>
```

A stream endpoint that can be used for reading data from a stream

#### Parameters:

| Parameter | C++ type | Description |
| :-------: | :-------: | ----------- |
| datatype | `Type` |  |

### LockType

An atomic synchronization element

Syntax: `!physical.lock`

Locks are atomic storage elements, which provides a primitive synchronization
mechanism that limits access to a resource.  A lock can be assigned a state,
and it can be acquired by only one user at a time when the acquiring state
matches the state stored in the lock.  The lock can be acquired again only
when the user release it with a state.

### OStreamType

A type specifiying a streaming endpoint for output

Syntax:

```
!physical.ostream<
  Type   # datatype
>
```

A stream endpoint that can be used for writing data to a stream

#### Parameters:

| Parameter | C++ type | Description |
| :-------: | :-------: | ----------- |
| datatype | `Type` |  |

### StreamDmaConnectType

A stream DMA connection

Syntax: `!physical.stream_dma_connect`

A stream DMA connection connects a buffer with a stream and perform DMA
operations between them.  A variable in this type can be passed as an
argument to another DMA connection as its successor.

### StreamFifoType

A streaming first-in first-out storage unit

Syntax:

```
!physical.stream_fifo<
  Type   # datatype
>
```

Fifos are first-in first-out storage elements, which takes an element in
as input and buffers it.  The elements in the fifo is sent to the output
whenever they are ready.  A fifo can have only one input connection and one
output connection.  The tag of the stream data will be preseved.

#### Parameters:

| Parameter | C++ type | Description |
| :-------: | :-------: | ----------- |
| datatype | `Type` |  |

### StreamHubType

A type specifiying a stream hub for broadcasting

Syntax:

```
!physical.stream_hub<
  Type   # datatype
>
```

A stream hub receives data from input streams and broadcast to output
stream.  The supported endpoint count of a stream hub is target-dependent.
If the data in a stream is optionally tagged for packet switching, only
the streams accepting the tags will receive the data.

#### Parameters:

| Parameter | C++ type | Description |
| :-------: | :-------: | ----------- |
| datatype | `Type` |  |
