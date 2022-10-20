//===- LayoutToPhysical.h -------------------------------------------------===//
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef PHY_CONVERSION_LAYOUT_TO_PHYSICAL_H_
#define PHY_CONVERSION_LAYOUT_TO_PHYSICAL_H_

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace phy {
std::unique_ptr<mlir::Pass> createLayoutToPhysical();
} // namespace phy

#endif // PHY_CONVERSION_LAYOUT_TO_PHYSICAL_H_
