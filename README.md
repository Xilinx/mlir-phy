# MLIR-based Abstract Physical Design
⏩ An MLIR dialect to express the mapping of PEs, buffers, networks and buses to an abstract physical floorplan.

![GitHub Workflow Status](https://img.shields.io/github/workflow/status/heterosys/mlir-phy/Build%20and%20Test)
![GitHub Pull Requests](https://img.shields.io/github/issues-pr-raw/heterosys/mlir-phy)

![](https://mlir.llvm.org//mlir-logo.png)

**WARNING:** This is a work-in-progress and will be actively changed.

This repository contains an [MLIR-based](https://mlir.llvm.org/) toolchain for expressing and generating the mapping of a task graph design on an abstract physical device.  This project is intended to be an intermediate representation between high-level tasks as in parallel programming, and their actual physical designs.

Documentations: [Design Principle](https://tinyurl.com/heterosys-mlir-phy), [Dialect Reference](https://heterosys.github.io/mlir-phy/PhyDialect.html), [Passes Reference](https://heterosys.github.io/mlir-phy/PhyPasses.html).

## How to Build

### 0. Clone MLIR-PHY and install prerequisites

```
git clone https://github.com/heterosys/mlir-phy.git
cd mlir-phy

sudo apt-get install -y build-essential python3-pip
pip3 install cmake ninja lit psutil
```

### 1. Install LLVM, Clang, and MLIR

You can download our nightly pre-built snapshot from https://github.com/heterosys/llvm-nightly.

```sh
OS_DISTRIBUTER=$(lsb_release -is | tr '[:upper:]' '[:lower:]')
OS_RELEASE=$(lsb_release -rs)
LLVM_VERSION="snapshot-20220706"
LLVM_URL="https://github.com/heterosys/llvm-nightly/releases/download/${LLVM_VERSION}/llvm-clang-mlir-dev-15.${OS_DISTRIBUTER}-${OS_RELEASE}.deb"

TEMP_DEB="$(mktemp)" && \
  wget -O "${TEMP_DEB}" ${LLVM_URL} && \
  (sudo dpkg -i "${TEMP_DEB}" || sudo apt-get -yf install)
rm -f "${TEMP_DEB}"
```

Please update the variable `LLVM_VERSION` according to `.github/scripts/install-build-deps.sh`.

Or alternatively, build LLVM from scratch.  When building manually, remember to enable `clang;mlir` in `-DLLVM_ENABLE_PROJECTS`.  `-DLLVM_BUILD_UTILS=ON` and `-DLLVM_INSTALL_UTILS=ON` shall be set as well.

### 2. Build MLIR-PHY

```sh
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Debug \
  -DLLVM_EXTERNAL_LIT=`which lit` \
  -DCMAKE_MAKE_PROGRAM=ninja -G Ninja
  # If you build llvm-project manually, add the following options:
  # -DLLVM_DIR=${absolute path to llvm-project}/build/lib/cmake/llvm
  # -DMLIR_DIR=${absolute path to llvm-project}/build/lib/cmake/mlir
cmake --build build --target all
```

To test MLIR-PHY:

```sh
cmake --build build --target check-phy
```

Cheers! 🍺
