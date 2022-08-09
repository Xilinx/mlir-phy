# RUN: bash -- %s phy-translate | FileCheck %s
# CHECK: OVERVIEW: MLIR-PHY translation tool

PHY_TRANSLATE=$1
${PHY_TRANSLATE} --help