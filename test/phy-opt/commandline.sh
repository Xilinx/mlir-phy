# RUN: bash -- %s phy-opt | FileCheck %s
# CHECK: OVERVIEW: MLIR-PHY optimizer driver
#
# This file is licensed under the MIT License.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: MIT
#
# Copyright (C) 2022, Advanced Micro Devices, Inc.

PHY_OPT=$1
${PHY_OPT} --help