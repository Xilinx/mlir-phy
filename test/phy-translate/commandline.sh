# RUN: bash -- %s phy-translate | FileCheck %s
# CHECK: OVERVIEW: MLIR-PHY translation tool
#
# This file is licensed under the MIT License.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2022, Advanced Micro Devices, Inc.

PHY_TRANSLATE=$1
${PHY_TRANSLATE} --help