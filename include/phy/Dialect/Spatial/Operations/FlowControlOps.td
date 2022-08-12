//===- FlowControlOps.td -----------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_DIALECT_SPATIAL_FLOW_CONTROL_OPS_TD
#define MLIR_PHY_DIALECT_SPATIAL_FLOW_CONTROL_OPS_TD

include "phy/Dialect/Spatial/SpatialBase.td"

//===----------------------------------------------------------------------===//
// ValidOp
//===----------------------------------------------------------------------===//

def ValidOp : Spatial_Op<"valid"> {
  let summary = "non-blocking test whether the queue is ready for dequeuing";
  let description = [{
    The `spatial.valid` operation tests if its operand queue has an element for
    dequeuing with the given tag.  This operation is non-blocking and returns
    a boolean result, which is true if there is an element to dequeue and the
    next element has the given tag.  Otherwise, the operation returns false.
    When `spatial.valid` returns true, `spatial.front` and `spatial.pop`
    return immediately without blocking.  If the 'tag' is not specified, tag
    #0 is used by default.

    Example:

    ```mlir
    %0 = spatial.valid<1>(%q : !spatial.queue<memref<i32>>)
    ```
  }];

  let arguments = (ins AnyQueueType:$queue,
                       OptionalAttr<I64Attr>:$tag);
  let results = (outs I1:$result);

  let assemblyFormat = [{
    (`<` $tag^ `>`)? `(` $queue `:` type($queue) `)` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// FullOp
//===----------------------------------------------------------------------===//

def FullOp : Spatial_Op<"full"> {
  let summary = "non-blocking test whether the queue is full";
  let description = [{
    The `spatial.full` operation tests its operand queue if the given queue is
    ready for enqueuing elements with the given tag.  This operation is
    non-blocking and returns a boolean result, which is true if it is full,
    or false if it is ready for enqueuing.  When `spatial.valid` returns true,
    `spatial.embrace` and `spatial.push` return immediately without blocking.  
    If the 'tag' is not specified, tag #0 is used by default.

    Example:

    ```mlir
    %0 = spatial.full<1>(%q : !spatial.queue<memref<i32>>)
    ```
  }];

  let arguments = (ins AnyQueueType:$queue,
                       OptionalAttr<I64Attr>:$tag);
  let results = (outs I1:$result);

  let assemblyFormat = [{
    (`<` $tag^ `>`)? `(` $queue `:` type($queue) `)` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// FrontOp
//===----------------------------------------------------------------------===//

def FrontOp : Spatial_Op<"front",
    [TypesMatchWith<"'queue' data has the same type as result",
                    "result", "queue",
                    "QueueType::get($_self.getContext(), $_self)">]> {

  let summary = "blocking retrieve the next data accessor";
  let description = [{
    The `spatial.front` operation retrieves the next, i.e., the oldest element
    with the given tag from its operand queue.  This operation is blocking and
    returns the retrieved data accessor as its result.  If the current next
    element does not have the given tag, or if the queue is currently empty,
    it blocks until the queue has a next element with the tag.  If the 'tag' is
    not specified, tag #0 is used by default.

    Example:

    ```mlir
    %queue = spatial.queue() : !spatial.flow<memref<i32>>
    %accessor = spatial.front<1>(%queue) : memref<i32>
    ```
  }];

  let arguments = (ins AnyQueueType:$queue,
                       OptionalAttr<I64Attr>:$tag);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    (`<` $tag^ `>`)? `(` $queue `)` attr-dict `:` type($result) 
  }];
}

//===----------------------------------------------------------------------===//
// EmplaceOp
//===----------------------------------------------------------------------===//

def EmplaceOp : Spatial_Op<"embrace",
    [TypesMatchWith<"'queue' data has the same type as result",
                    "result", "queue",
                    "QueueType::get($_self.getContext(), $_self)">]> {

  let summary = "blocking construct the last data accessor";
  let description = [{
    The `spatial.embrace` operation allocate a memory space at the last of the
    queue for preparing an element to enqueue.  The element is tagged with the
    given tag, or tagged #0 by default.  When the queue has no space to be
    allocated, this operation blocks until it is available.  It returns the
    accessor to the last element for preparing the data as its result.

    Example:

    ```mlir
    %queue = spatial.queue() : !spatial.flow<memref<i32>>
    %accessor = spatial.embrace<1>(%queue) : memref<i32>
    ```
  }];

  let arguments = (ins AnyQueueType:$queue,
                       OptionalAttr<I64Attr>:$tag);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    (`<` $tag^ `>`)? `(` $queue `)` attr-dict `:` type($result) 
  }];
}

//===----------------------------------------------------------------------===//
// PopOp
//===----------------------------------------------------------------------===//

def PopOp : Spatial_Op<"pop"> {
  let summary = "blocking remove the next element in the queue";
  let description = [{
    The `spatial.pop` operation removes the next, i.e., the oldest element with
    the given tag in the queue, whose accessor can be retrieved by calling
    `front`.  When there is no element in the queue tagged with the given tag,
    or if the next element is not tagged with the given tag, this function
    blocks and it waits until such an element is ready for dequeuing. If the
    'tag' is not specified, tag #0 is used by default.

    Example:

    ```mlir
    spatial.pop<1>(%queue: !spatial.flow<memref<i32>>)
    ```
  }];

  let arguments = (ins AnyQueueType:$queue,
                       OptionalAttr<I64Attr>:$tag);
  
  let assemblyFormat = [{
    (`<` $tag^ `>`)? `(` $queue `:` type($queue) `)` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// PushOp
//===----------------------------------------------------------------------===//

def PushOp : Spatial_Op<"push"> {
  let summary = "commit the last element to the queue";
  let description = [{
    The `spatial.push` operation commits the last `spatial.emplace` constructed
    element with the given tag to the queue.  The enqueued element becomes the
    new last element in the queue.  If the 'tag' is not specified, tag #0 is used
    by default.
    
    When there is no element allocated by `spatial.emplace` with the given tag,
    this operation implicitly calls the corresponding `spatial.emplace` right
    before it, with the element content undefined.  In this case, the operation
    is blocking.  This behavior is useful when !spatial.flow<none> is used.

    Example:

    ```mlir
    spatial.push<1>(%queue: !spatial.flow<memref<i32>>)
    ```
  }];

  let arguments = (ins AnyQueueType:$queue,
                       OptionalAttr<I64Attr>:$tag);
  
  let assemblyFormat = [{
    (`<` $tag^ `>`)? `(` $queue `:` type($queue) `)` attr-dict
  }];
}

#endif  // MLIR_PHY_DIALECT_SPATIAL_FLOW_CONTROL_OPS_TD
