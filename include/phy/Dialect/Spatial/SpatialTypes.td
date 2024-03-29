//===- SpatialTypes.td -------------------------------------*- tablegen -*-===//
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_DIALECT_SPATIAL_TYPES_TD
#define MLIR_PHY_DIALECT_SPATIAL_TYPES_TD

include "phy/Dialect/Spatial/SpatialBase.td"

//===----------------------------------------------------------------------===//
// NodeType
//===----------------------------------------------------------------------===//

def NodeType : Spatial_VertexType<"Node", "node"> {
  let summary = "A computing element";
  let description = [{
    Nodes are computing elements.  Each node contains a software function
    free-running since system startup.  The function interacts with queues
    connected to the node with flows.
  }];
}

//===----------------------------------------------------------------------===//
// QueueType
//===----------------------------------------------------------------------===//

def QueueType : Spatial_VertexType<"Queue", "queue"> {
  let summary = "A storage elements to be connected by nodes";
  let description = [{
    Queues are storage elements, which host a message queue for the connected
    nodes.  A node allocates space in the queue to enqueue, or accesses the
    space to dequeue.  An element can be tagged with a number.  When multiple
    nodes contend to enqueue, one node is selected.  An element is dequeued
    when all nodes observing the tag dequeue it.
  }];

  let parameters = (ins "Type":$datatype);
  let assemblyFormat = "`<` $datatype `>`";
}

class QueueTypeOf<Type element> : Type<
  And<[
    CPred<"$_self.isa<phy::spatial::QueueType>()">,
    SubstLeaves<
      "$_self",
      "$_self.cast<phy::spatial::QueueType>().getDatatype()",
      element.predicate>,
  ]>,
  "a queue of " # element.summary
>;

def AnyQueueType : Type<
  Or<[
    QueueTypeOf<AnyStaticShapeMemRef>.predicate,
    QueueTypeOf<NoneType>.predicate,
  ]>,
  "a queue of static shape memref or none"
>;

//===----------------------------------------------------------------------===//
// PromiseType
//===----------------------------------------------------------------------===//

def PromiseType : Spatial_Type<"Promise", "promise"> {
  let summary = "A type specifiying a on-going memory access";
  let description = [{
    A non-blocking memory access returns a promise, which may be waited to get
    the access result.
  }];

  let parameters = (ins "Type":$datatype);
  let assemblyFormat = "`<` $datatype `>`";
}

class PromiseTypeOf<Type element> : Type<
  And<[
    CPred<"$_self.isa<phy::spatial::PromiseType>()">,
    SubstLeaves<
      "$_self",
      "$_self.cast<phy::spatial::PromiseType>().getDatatype()",
      element.predicate>
  ]>,
  "a promise of " # element.summary
>;

def AnyPromiseType : Type<
  Or<[
    PromiseTypeOf<AnyType>.predicate,
  ]>,
  "a promise"
>;

//===----------------------------------------------------------------------===//
// Type Constraints
//===----------------------------------------------------------------------===//

def AnyVertexType : Type<
  Or<[
    AnyQueueType.predicate,
    NodeType.predicate,
  ]>,
  "a queue or a node"
>;

#endif  // MLIR_PHY_DIALECT_SPATIAL_TYPES_TD
