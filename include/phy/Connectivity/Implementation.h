//===- Implementation.h -----------------------------------------*- C++ -*-===//
//
// This file defines the implementation of a virtual resource.
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_CONNECTIVITY_IMPLEMENTATION_H
#define MLIR_PHY_CONNECTIVITY_IMPLEMENTATION_H

#include "phy/Connectivity/Resource.h"

#include <list>
#include <map>
#include <string>

namespace phy {
namespace connectivity {

class Implementation {
  /**
   * A list of phys to implement a virtual resource.
   *
   * To serialize the list into a string, the phys are concatenated
   * seperated by a comma. For example, "tile/7.0/buffer,tile/7.0/lock".
   */
  const std::string delim = ",";

public:
  std::list<PhysicalResource> phys;

  Implementation() {}
  Implementation(std::string serialized);
  Implementation(std::list<PhysicalResource> phys) : phys(phys) {}
  std::string toString();
};

} // namespace connectivity
} // namespace phy

#endif // MLIR_PHY_CONNECTIVITY_IMPLEMENTATION_H
