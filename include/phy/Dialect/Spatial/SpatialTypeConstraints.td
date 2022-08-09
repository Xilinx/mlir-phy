//===- SpatialTypeConstraints.td ---------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_DIALECT_SPATIAL_TYPE_CONSTRAINTS_TD
#define MLIR_PHY_DIALECT_SPATIAL_TYPE_CONSTRAINTS_TD

include "phy/Dialect/Spatial/SpatialBase.td"
include "phy/Dialect/Spatial/SpatialTypes.td"

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

class FlowTypeOf<Type element> : Type<
  And<[
    CPred<"$_self.isa<phy::spatial::FlowType>()">,
    SubstLeaves<
      "$_self",
      "$_self.cast<phy::spatial::FlowType>().getDatatype()",
      element.predicate>,
  ]>,
  "a flow of " # element.summary
>;

class HandleTypeOf<Type element> : Type<
  And<[
    CPred<"$_self.isa<phy::spatial::HandleType>()">,
    SubstLeaves<
      "$_self",
      "$_self.cast<phy::spatial::HandleType>().getDatatype()",
      element.predicate>
  ]>,
  "a handle of " # element.summary
>;

def AnyQueueType : Type<
  Or<[
    QueueTypeOf<AnyStaticShapeMemRef>.predicate,
    QueueTypeOf<NoneType>.predicate,
  ]>,
  "a queue of static shape memref or none"
>;

def AnyFlowType : Type<
  Or<[
    FlowTypeOf<AnyStaticShapeMemRef>.predicate,
    FlowTypeOf<NoneType>.predicate,
  ]>,
  "a flow of static shape memref or none"
>;

def AnyVertexType : Type<
  Or<[
    AnyQueueType.predicate,
    NodeType.predicate,
  ]>,
  "a queue or a node"
>;

def AnyEdgeType : Type<
  Or<[
    AnyFlowType.predicate,
  ]>,
  "a flow"
>;

#endif  // MLIR_PHY_DIALECT_SPATIAL_TYPE_CONSTRAINTS_TD
