##===- install-build-deps.sh - GitHub CI scripts -----------------*- sh -*-===//
##
## This file is licensed under the MIT License.
## SPDX-License-Identifier: MIT
##
## Copyright (C) 2022, Advanced Micro Devices, Inc.
##
##===----------------------------------------------------------------------===//
#!/bin/sh
set -e

sudo apt-get update
sudo apt-get purge -y libgcc-*-dev || true
sudo apt-get install -y build-essential python3-pip
sudo apt-get autoremove -y

sudo -H python3 -m pip install --upgrade pip==20.3.4
sudo -H python3 -m pip install cmake
sudo -H python3 -m pip install ninja
sudo -H python3 -m pip install lit
sudo -H python3 -m pip install psutil

# install the selected llvm-clang-mlir-dev snapshot
OS_DISTRIBUTER=$(lsb_release -is | tr '[:upper:]' '[:lower:]')
OS_RELEASE=$(lsb_release -rs)
LLVM_VERSION="snapshot-20220926"
LLVM_URL="https://github.com/heterosys/llvm-nightly/releases/download/${LLVM_VERSION}/llvm-clang-mlir-dev-15.${OS_DISTRIBUTER}-${OS_RELEASE}.deb"
TEMP_DEB="$(mktemp)" && wget -O "${TEMP_DEB}" ${LLVM_URL} && (sudo dpkg -i "${TEMP_DEB}" || sudo apt-get -yf install)
rm -f "${TEMP_DEB}"

AIE_URL="https://github.com/heterosys/mlir-aie/releases/latest/download/mlir-aie-${OS_DISTRIBUTER}-${OS_RELEASE}.deb"
TEMP_DEB="$(mktemp)" && wget -O "${TEMP_DEB}" ${AIE_URL} && (sudo dpkg -i "${TEMP_DEB}" || sudo apt-get -yf install)
rm -f "${TEMP_DEB}"