//===- SpatialTypes.td -------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_DIALECT_SPATIAL_TYPES_TD
#define MLIR_PHY_DIALECT_SPATIAL_TYPES_TD

include "phy/Dialect/Spatial/SpatialBase.td"

include "mlir/IR/AttrTypeBase.td"

class Spatial_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Spatial_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

class Spatial_VertexType<string name, string typeMnemonic, list<Trait> traits = []>
    : Spatial_Type<name, typeMnemonic, traits> {
  let mnemonic = typeMnemonic;
}

class Spatial_EdgeType<string name, string typeMnemonic, list<Trait> traits = []>
    : Spatial_Type<name, typeMnemonic, traits> {
  let mnemonic = typeMnemonic;
}

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

//===----------------------------------------------------------------------===//
// FlowType
//===----------------------------------------------------------------------===//

def FlowType : Spatial_EdgeType<"Flow", "flow"> {
  let summary = "A directed connection between a node and a queue";
  let description = [{
    Flows are implicitly created between nodes and queues when a queue is
    passed to a node function.  A flow connects a node vertex with a queue
    vertex.  A flow is directed.
  }];

  let parameters = (ins "Type":$datatype);
  let assemblyFormat = "`<` $datatype `>`";
}

//===----------------------------------------------------------------------===//
// HandleType
//===----------------------------------------------------------------------===//

def HandleType : Spatial_Type<"Handle", "handle"> {
  let summary = "A type specifiying a on-going memory access";
  let description = [{
    A non-blocking memory access returns a handle, which may be waited to get
    the access result.
  }];

  let parameters = (ins "Type":$datatype);
  let assemblyFormat = "`<` $datatype `>`";
}

#endif  // MLIR_PHY_DIALECT_SPATIAL_TYPES_TD
