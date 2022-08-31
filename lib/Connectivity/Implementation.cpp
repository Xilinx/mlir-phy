//===- Implementation.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "phy/Connectivity/Implementation.h"

#include "phy/Connectivity/Implementation/Buffer.h"

#include "mlir/IR/Builders.h"

using namespace phy::connectivity;

std::shared_ptr<Implementation>
phy::connectivity::ImplementationFactory(PhysicalResource phy,
                                         ImplementationContext &context) {
  if (phy.key == "buffer") {
    return std::make_shared<BufferImplementation>(phy, context);
  } else {
    return nullptr;
  }
}

void Implementation::attachMetadata() {
  auto builder = mlir::OpBuilder::atBlockEnd(context.module.getBody());

  for (auto metadata : phy.metadata) {
    std::string attr_name = context.device + "." + metadata.first;
    cached_op->setAttr(attr_name, builder.getStringAttr(metadata.second));
  }
}

mlir::Operation *Implementation::getOperation() {
  if (!cached_op) {
    cached_op = this->createOperation();
    attachMetadata();
  }
  return cached_op;
}

std::shared_ptr<Implementation>
ImplementationContext::getImplementation(PhysicalResource phy) {
  auto identifier = phy.toString();

  if (!impls.count(identifier)) {
    impls[identifier] = ImplementationFactory(phy, *this);
  }
  return impls[identifier];
}
