//===- Graph.h --------------------------------------------------*- C++ -*-===//
//
// This file defines the basic components on a connectivity graph.
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_CONNECTIVITY_GRAPH_H
#define MLIR_PHY_CONNECTIVITY_GRAPH_H

#include <list>
#include <map>
#include <string>

namespace phy {
namespace connectivity {

class Phy {
  /**
   * A phy contains two pieces of information: implementation method (key) and
   * target information (metadata).  The implementation method is
   * target-independent, for example, "buffer".  The target information is
   * target-dependent and is directly handled by the lowering passes as
   * attribute strings.  It is represented as a string map.  For example,
   * "tile" = "7.0".
   *
   * To serialize the phy into a string, the target information is
   * concatenated first seperated by a slash.  The implementation method
   * follows with a slash seperated.  For example, "tile/7.0/buffer".
   */

  const std::string delim = "/";

public:
  std::string key;
  std::map<std::string, std::string> metadata;

  Phy() {}
  Phy(std::string slash_seperated_info);
  Phy(std::string key, std::map<std::string, std::string> metadata)
      : key(key), metadata(metadata) {}
  std::string toString();
};

class Slot {
  /**
   * A list of phys to implement a spatial vertex.
   *
   * To serialize the list into a string, the phys are concatenated
   * seperated by a comma. For example, "tile/7.0/lock,tile/7.0/core_affinity".
   */
  const std::string delim = ",";

public:
  std::list<Phy> phys;

  Slot() {}
  Slot(std::string comma_seperated_phys);
  Slot(std::list<Phy> phys) : phys(phys) {}
  std::string toString();
};

} // namespace connectivity
} // namespace phy

#endif // MLIR_PHY_CONNECTIVITY_GRAPH_H
