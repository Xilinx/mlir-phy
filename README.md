# MLIR-based Physical Design
⏩ MLIR toolchain to design and map computing nodes and message queues to a physical design.

![](https://mlir.llvm.org//mlir-logo.png)

**WARNING:** This is a work-in-progress and will be actively changed.

This repository contains an [MLIR-based](https://mlir.llvm.org/) toolchain that maps a task graph design onto abstract physical devices.  It maps target-independent descriptions to actual physical implementations.  This project is intended to be an intermediate representation between high-level tasks as in parallel programming, and their actual physical designs.  Three dialects are defined to facilitate the lowering from logical spatial design into a physical target:

- `spatial`:  a target-independent, stateful and free-running description of a spatial design.
- `layout`: a target-dependent mapping of `spatial` onto the devices and platforms.
- `physical`: a unified abstract layer over the target-dependent dialects that provides direct access to low-level features.

Please see our detailed design at [Design Principle and Examples](https://tinyurl.com/heterosys-mlir-phy).

## Documentations

|            | Dialect References | Passes References |
| ---------- | ------------------ | ----------------- |
| `spatial`  | [Dialect](https://heterosys.github.io/mlir-phy/SpatialDialect.html) | [Passes](https://heterosys.github.io/mlir-phy/SpatialPasses.html) |
| `layout`   | [Dialect](https://heterosys.github.io/mlir-phy/LayoutDialect.html) | [Passes](https://heterosys.github.io/mlir-phy/LayoutPasses.html) |
| `physical` | [Dialect](https://heterosys.github.io/mlir-phy/PhysicalDialect.html) | [Passes](https://heterosys.github.io/mlir-phy/PhysicalPasses.html) |
| Conversion | N/A | [Passes](https://heterosys.github.io/mlir-phy/PhyConversionPass.html) |

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
