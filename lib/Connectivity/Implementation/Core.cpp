//===- Core.cpp -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "phy/Connectivity/Implementation/Core.h"

#include "mlir/IR/Builders.h"
#include "mlir/Support/LLVM.h"

#include <set>

using namespace mlir;
using namespace phy;
using namespace phy::connectivity;

mlir::Operation *CoreImplementation::createOperation() {
  assert(node.getOperation() && "a core must be associated with a node");

  auto builder = OpBuilder::atBlockEnd(context.module.getBody());
  return builder.create<physical::CoreOp>(
      builder.getUnknownLoc(), physical::CoreType::get(builder.getContext()),
      translateFunction(), translateOperands());
}

llvm::SmallVector<mlir::Value>
CoreImplementation::getOperandValues(mlir::Value operand) {
  llvm::SmallVector<Value> values;

  auto queue = dyn_cast<spatial::QueueOp>(operand.getDefiningOp());
  assert(queue && "operand is a defined queue");

  for (auto impl : queue_impls[queue]) {
    auto impl_op = impl.lock()->getOperation();
    assert(impl_op->getNumResults() == 1 && "returns one value");
    auto value = impl_op->getResult(0);
    values.push_back(value);
  }

  return values;
}

llvm::SmallVector<Value> CoreImplementation::translateOperands() {
  llvm::SmallVector<Value> translated;

  for (auto operand : node.operands())
    for (auto value : getOperandValues(operand))
      translated.push_back(value);

  return translated;
}

mlir::StringRef CoreImplementation::translateFunction() {

  auto original_op = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
      node, StringAttr::get(node.getContext(), node.callee()));
  assert(original_op && "function must be defined");
  auto original_fn_type = original_op.getFunctionType();
  int original_num_inputs = original_fn_type.getNumInputs();

  // Clone function
  OpBuilder builder(original_op);
  auto translated_op =
      dyn_cast<func::FuncOp>(builder.clone(*(original_op.getOperation())));
  assert(translated_op && "function cloned");

  // Set name to translated name
  auto tranlated_name = context.getUniqueSymbol(node.callee(), node);
  translated_op.setName(tranlated_name);

  // Clone function type
  llvm::SmallVector<Type> translated_inputs;
  for (int i = 0; i < original_num_inputs; i++)
    translated_inputs.push_back(original_fn_type.getInput(i));

  // Build new function arguments
  for (int i = 0; i < original_num_inputs; i++)
    for (auto value : getOperandValues(node.getOperand(i))) {
      translated_inputs.push_back(value.getType());
      translated_op.getCallableRegion()->addArgument(value.getType(),
                                                     translated_op.getLoc());
    }
  translated_op.setType(builder.getFunctionType(translated_inputs, {}));

  // Translate the usage of function arguments
  std::set<Operation *> users_to_be_translated;
  for (auto arg : translated_op.getArguments())
    for (auto user : arg.getUsers())
      users_to_be_translated.insert(user);
  for (auto user : users_to_be_translated) {
    user->erase();
  }

  // Erase original arguments
  for (int idx = original_num_inputs - 1; idx >= 0; idx--)
    translated_op.eraseArgument(idx);

  return tranlated_name;
}

void CoreImplementation::addPredecessor(std::weak_ptr<Implementation> pred,
                                        mlir::Operation *src,
                                        mlir::Operation *dest) {
  if (auto queue = dyn_cast<spatial::QueueOp>(src))
    addQueueImpl(queue, pred);
}

void CoreImplementation::addSuccessor(std::weak_ptr<Implementation> succ,
                                      mlir::Operation *src,
                                      mlir::Operation *dest) {
  if (auto queue = dyn_cast<spatial::QueueOp>(dest))
    addQueueImpl(queue, succ);
}

void CoreImplementation::addSpatialOperation(mlir::Operation *spatial) {
  if (auto node_op = dyn_cast<spatial::NodeOp>(spatial)) {
    assert(!node.getOperation() && "a core can only hold a node");
    node = node_op;
  } else {
    assert("a core can only implement a node");
  }
}

void CoreImplementation::addQueueImpl(spatial::QueueOp queue,
                                      std::weak_ptr<Implementation> impl) {
  queue_impls[queue].push_back(impl);
}
