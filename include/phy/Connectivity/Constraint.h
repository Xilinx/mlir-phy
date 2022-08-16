//===- Constraint.h ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_CONNECTIVITY_CONSTRAINT_H
#define MLIR_PHY_CONNECTIVITY_CONSTRAINT_H

#include <map>
#include <string>

namespace phy {
namespace connectivity {

/**
 * A capacity is the resource constraints of a phy.  When a vertex occupies a
 * phy, the resource utilization of the vertex occupies the capacity of all its
 * phys.  For example, if a {"count": 1} vertex occupies a buffer and a lock,
 * both the buffer and the lock's count is reduced by 1..
 *
 * A capacity lives between spatial and layout.
 */
using Capacity = std::map<std::string, int>;

/**
 * A target support is the implementation constraints of a phy.  For example, if
 * a {"states": 2} lock is implemented, only lock_acquire<0> and lock_acquire<1>
 * will be used.  Another example, a {"width_bytes": 32} stream limits the
 * physical implementation to use this width.
 *
 * A target support lives between layout and physical.
 */
using TargetSupport = std::map<std::string, int>;

} // namespace connectivity
} // namespace phy

#endif // MLIR_PHY_CONNECTIVITY_CONSTRAINT_H
