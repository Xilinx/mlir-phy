//===- phy-translate.cpp - PHY Translation Driver ---------------------===//
//
// This is a command line utility that translates a file from/to MLIR using one
// of the registered translations.
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

int main(int argc, char **argv) {
  return mlir::failed(
      mlir::mlirTranslateMain(argc, argv, "MLIR-PHY translation tool"));
}
