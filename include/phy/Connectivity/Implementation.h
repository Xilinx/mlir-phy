//===- Implementation.h -----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_CONNECTIVITY_IMPLEMENTATION_H
#define MLIR_PHY_CONNECTIVITY_IMPLEMENTATION_H

#include "phy/Connectivity/Resource.h"

#include "phy/Connectivity/ResourceList.h"
#include "phy/Dialect/Physical/PhysicalDialect.h"

#include <map>
#include <memory>
#include <string>

namespace phy {
namespace connectivity {

class Implementation;
class ImplementationContext;

// The factory takes a physical resource and returns an implementation object.
std::shared_ptr<Implementation>
ImplementationFactory(PhysicalResource phy, ImplementationContext &context);

/**
 * Implementation of a physical dialect operation for a physical resource.
 */
class Implementation {
protected:
  PhysicalResource phy;
  ImplementationContext &context;
  mlir::Operation *implemented_op;

  // This is the function to be overrided by children classes to create the
  // actual implementation operation.  The operation returned by this method
  // will automatically have the metadata in the physical resource attached.
  virtual mlir::Operation *createOperation() = 0;

  // This method attach the metadata in the physical resource to the
  // implemented_op.
  void attachMetadata();

public:
  Implementation(PhysicalResource phy, ImplementationContext &context)
      : phy(phy), context(context), implemented_op(nullptr) {}

  //===---------------------- Neighbor Information ------------------------===//
  // The implementer will call the methods to notify the object of its
  // siblings, predecessors, and successors.  For example, on a route of
  // place<"buffer,lock">, route<["dma", "stream"]>, the predecessors of "dma"
  // are "buffer" and "lock", the sibling of "lock" is "buffer" and itself, the
  // successor of "lock" is "dma", etc.  Another example: for a null route of
  // place<"buffer,lock">, route<>, place<"node">, the predecessors of "node"
  // are "buffer" and "lock". An implementation shall override those methods
  // if the information is relevant.
  virtual void addSibling(std::weak_ptr<Implementation> sib){};
  virtual void addPredecessor(std::weak_ptr<Implementation> pred,
                              mlir::Operation *src, mlir::Operation *dest){};
  virtual void addSuccessor(std::weak_ptr<Implementation> succ,
                            mlir::Operation *src, mlir::Operation *dest){};
  //===---------------------- Neighbor Information ------------------------===//

  //===--------------------- Implementee Information ----------------------===//
  // The implementer will call the methods to notify the object of what spatial
  // operation or flow is to be implemented.  An implementation shall override
  // those methods if the information is relevant.
  virtual void addSpatialOperation(mlir::Operation *spatial) {}
  virtual void addSpatialFlow(mlir::Operation *src, mlir::Operation *dest) {}
  //===--------------------- Implementee Information ----------------------===//

  //===--------------------------- Translation ----------------------------===//
  // The implementer, or the implementation's siblings, predecessors or
  // successors may call this function to get the implemented operation in
  // the physical dialect.  This function is 'cached', meaning, only the
  // first invocation should create a new operation, while the following
  // invocations must return the same pointer as the first fall.  This is to
  // provide an automatic recursive resolving of the dependencies.  The
  // implementer is guaranteed to call each implementation once.
  mlir::Operation *getOperation();

  // The implementation's predecessors or successors may call this function to
  // translate the spatial usage operations, e.g. spatial.emplace, spatial.pop,
  // etc, into the corresponding implementation's operation.  This method shall
  // not erase the translated operation as each operation might require
  // multiple implementations.  However, the result of the translated operation
  // shall be redirected by one of the operations.
  virtual void translateUserOperation(mlir::Value value,
                                      mlir::Operation *user){};
};

class ImplementationContext {
  std::map<llvm::StringRef, int> unique_suffix;

public:
  std::string device;
  std::map<mlir::Operation *, std::list<std::weak_ptr<Implementation>>>
      placements;
  std::map<std::string, std::shared_ptr<Implementation>> impls;
  mlir::ModuleOp module;

  ImplementationContext(mlir::ModuleOp module, std::string device)
      : device(device), module(module) {}

  void place(mlir::Operation *spatial, ResourceList resources);
  void route(mlir::Operation *src, mlir::Operation *dest,
             std::list<ResourceList> resources);

  void implementAll();

  mlir::StringAttr getUniqueSymbol(llvm::StringRef base, mlir::Operation *op);
};

} // namespace connectivity
} // namespace phy

#endif // MLIR_PHY_CONNECTIVITY_IMPLEMENTATION_H
