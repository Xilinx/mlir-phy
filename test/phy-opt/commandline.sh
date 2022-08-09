# RUN: bash -- %s phy-opt | FileCheck %s
# CHECK: OVERVIEW: MLIR-PHY optimizer driver

PHY_OPT=$1
${PHY_OPT} --help