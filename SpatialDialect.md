<!-- Autogenerated by mlir-tblgen; don't manually edit -->
# 'spatial' Dialect

This dialect defines a graph of stateful vertices of free-running compute
nodes and message queues.  The vertices are connected using flow control 
edges called flows.  This dialect is intended to be target-independent
and express the logical concurrency semantics.  It does not express
anything that is target-dependent.

[TOC]

## Operation definition

### `spatial.bridge` (::phy::spatial::BridgeOp)

create a compute node that connectes two queues


Syntax:

```
operation ::= `spatial.bridge` `(` $src `->` $dest `:` type($dest) `)` attr-dict
```

The `spatial.bridge` operation represents a creation of a bridge, which is
a simple special node, which bridges two queues, one input, and one output.
A bridge, whenever the output queue is not full and the input queue is not
empty, reads the data from the input queue and writes the data into the
output queue.

Example:

```mlir
%bridge = spatial.bridge(%q1 -> %q2: !spatial.queue<memref<8xi32>>)
```

Traits: SameTypeOperands

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `src` | a queue of static shape memref or none
| `dest` | a queue of static shape memref or none

#### Results:

| Result | Description |
| :----: | ----------- |
| `node` | A computing element

### `spatial.emplace` (::phy::spatial::EmplaceOp)

blocking construct the last data accessor


Syntax:

```
operation ::= `spatial.emplace` `(` $queue `)` attr-dict `:` type($result)
```

The `spatial.emplace` operation allocate a memory space at the last of the
queue for preparing an element to enqueue.  When the queue has no space to
be allocated, this operation blocks until it is available.  It returns the
accessor to the last element for preparing the data as its result.

Example:

```mlir
%queue = spatial.queue() : !spatial.queue<memref<i32>>
%accessor = spatial.emplace(%queue) : memref<i32>
```

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `queue` | a queue of static shape memref or none

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | any type

### `spatial.front` (::phy::spatial::FrontOp)

blocking retrieve the next data accessor


Syntax:

```
operation ::= `spatial.front` `(` $queue `)` attr-dict `:` type($result)
```

The `spatial.front` operation retrieves the next, i.e., the oldest element
from its operand queue.  This operation is blocking and returns the
retrieved data accessor as its result.

Example:

```mlir
%queue = spatial.queue() : !spatial.queue<memref<i32>>
%accessor = spatial.front(%queue) : memref<i32>
```

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `queue` | a queue of static shape memref or none

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | any type

### `spatial.full` (::phy::spatial::FullOp)

non-blocking test whether the queue is full


Syntax:

```
operation ::= `spatial.full` `(` $queue `:` type($queue) `)` attr-dict
```

The `spatial.full` operation tests its operand queue if the given queue is
ready for enqueuing elements.  This operation is non-blocking and returns a
boolean result, which is true if it is full, or false if it is ready for
enqueuing.  When `spatial.valid` returns true, `spatial.emplace` and
`spatial.push` return immediately without blocking.

Example:

```mlir
%0 = spatial.full(%q : !spatial.queue<memref<i32>>)
```

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `queue` | a queue of static shape memref or none

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | 1-bit signless integer

### `spatial.node` (::phy::spatial::NodeOp)

compute node creation operation


Syntax:

```
operation ::= `spatial.node` $callee `(` $operands `)` attr-dict `:` functional-type($operands, $node)
```

The `spatial.node` operation represents a creation of a logical compute node
that has the function argument as its entry point.  Each node invokes the
software function `callee` free-running since system startup.  The function
interacts with queues connected to the node as in `operands`.  The function
must be within the same symbol scope as the operation.  The `operands` must
match the specified function type.

Example:

```mlir
%node = spatial.node @func(%queue)
      : (!spatial.queue<memref<1024xi32>>) -> !spatial.node
```

Interfaces: CallOpInterface, SymbolUserOpInterface

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `callee` | ::mlir::FlatSymbolRefAttr | flat symbol reference attribute

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `operands` | a queue of static shape memref or none

#### Results:

| Result | Description |
| :----: | ----------- |
| `node` | A computing element

### `spatial.pop` (::phy::spatial::PopOp)

blocking remove the next element in the queue


Syntax:

```
operation ::= `spatial.pop` `(` $queue `:` type($queue) `)` attr-dict
```

The `spatial.pop` operation removes the next, i.e., the oldest element in
the queue, whose accessor can be retrieved by calling `front`.

Example:

```mlir
spatial.pop(%queue: !spatial.queue<memref<i32>>)
```

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `queue` | a queue of static shape memref or none

### `spatial.push` (::phy::spatial::PushOp)

commit the last element to the queue


Syntax:

```
operation ::= `spatial.push` `(` $queue `:` type($queue) `)` attr-dict
```

The `spatial.push` operation commits the last `spatial.emplace` constructed
element to the queue.  The enqueued element becomes the new last element in
the queue.

When there is no element allocated by `spatial.emplace`, this operation
implicitly calls the corresponding `spatial.emplace` right before it, with
the element content undefined.  In this case, the operation is blocking.
This behavior is useful when !spatial.queue<none> is used.

Example:

```mlir
spatial.push(%queue: !spatial.queue<memref<i32>>)
```

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `queue` | a queue of static shape memref or none

### `spatial.queue` (::phy::spatial::QueueOp)

message queue creation operation


Syntax:

```
operation ::= `spatial.queue` `<` $depth `>` `(` `)` attr-dict `:` type($queue)
```

The `spatial.queue` operation represents a creation of a queue that has
the type argument as its element's datatype.  The created queue can be used
as an operand in `spatial.node` operations for the software function to
access and communicate.

Example:

```mlir
%queue = spatial.queue<2>(): !spatial.queue<memref<16xi32>>
```

#### Attributes:

| Attribute | MLIR Type | Description |
| :-------: | :-------: | ----------- |
| `depth` | ::mlir::IntegerAttr | 64-bit signless integer attribute

#### Results:

| Result | Description |
| :----: | ----------- |
| `queue` | a queue of static shape memref or none

### `spatial.start_load` (::phy::spatial::StartLoadOp)

non-blocking load of the data


Syntax:

```
operation ::= `spatial.start_load` $memref `[` $indices `]` attr-dict `:` type($memref)
```

A non-blocking bus access that reads the data from the memory in a queue
as specified in 'memref'.  This operation returns a promise, which may be
waited using `spatial.wait` to get the access result.

Example:

```mlir
%load_promise = spatial.start_load %mem[%idx] : memref<1024xi32>
%0 = spatial.wait(%load_promise) : i32
```

Traits: MemRefsNormalizable

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `memref` | statically shaped memref of any type values
| `indices` | index

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | any type

### `spatial.start_store` (::phy::spatial::StartStoreOp)

non-blocking store of the data


Syntax:

```
operation ::= `spatial.start_store` $value `,` $memref `[` $indices `]` attr-dict `:` type($memref)
```

A non-blocking bus access that stores the data to a bufferin a queue
as specified in 'memref'.  This operation returns a promise, which may
be waited using `spatial.wait`.

Example:

```mlir
%store_promise = spatial.start_store %0, %mem[%idx] : memref<1024xi32>
spatial.wait(%store_promise) : none
```

Traits: MemRefsNormalizable

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `value` | any type
| `memref` | statically shaped memref of any type values
| `indices` | index

#### Results:

| Result | Description |
| :----: | ----------- |
| `promise` | a promise of none type

### `spatial.valid` (::phy::spatial::ValidOp)

non-blocking test whether the queue is ready for dequeuing


Syntax:

```
operation ::= `spatial.valid` `(` $queue `:` type($queue) `)` attr-dict
```

The `spatial.valid` operation tests if its operand queue has an element for
dequeuing.  This operation is non-blocking and returns a boolean result,
which is true if there is an element to dequeue.  Otherwise, the operation
returns false.  When `spatial.valid` returns true, `spatial.front` and
`spatial.pop` return immediately without blocking.

Example:

```mlir
%0 = spatial.valid(%q : !spatial.queue<memref<i32>>)
```

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `queue` | a queue of static shape memref or none

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | 1-bit signless integer

### `spatial.wait` (::phy::spatial::WaitOp)

blocking wait until a promise is ready


Syntax:

```
operation ::= `spatial.wait` `(` $promise `)` attr-dict `:` type($result)
```

A non-blocking bus access returns a promise, which may be waited using this
operation to get the access result.  For store access, none is returned.

Example:

```mlir
%0 = spatial.wait(%load_promise) : i32
spatial.wait(%store_promise) : none
```

#### Operands:

| Operand | Description |
| :-----: | ----------- |
| `promise` | a promise

#### Results:

| Result | Description |
| :----: | ----------- |
| `result` | any type

## Type definition

### NodeType

A computing element

Syntax: `!spatial.node`

Nodes are computing elements.  Each node contains a software function
free-running since system startup.  The function interacts with queues
connected to the node with flows.

### PromiseType

A type specifiying a on-going memory access

Syntax:

```
!spatial.promise<
  Type   # datatype
>
```

A non-blocking memory access returns a promise, which may be waited to get
the access result.

#### Parameters:

| Parameter | C++ type | Description |
| :-------: | :-------: | ----------- |
| datatype | `Type` |  |

### QueueType

A storage elements to be connected by nodes

Syntax:

```
!spatial.queue<
  Type   # datatype
>
```

Queues are storage elements, which host a message queue for the connected
nodes.  A node allocates space in the queue to enqueue, or accesses the
space to dequeue.  An element can be tagged with a number.  When multiple
nodes contend to enqueue, one node is selected.  An element is dequeued
when all nodes observing the tag dequeue it.

#### Parameters:

| Parameter | C++ type | Description |
| :-------: | :-------: | ----------- |
| datatype | `Type` |  |
