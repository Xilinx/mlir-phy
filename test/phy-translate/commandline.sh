# RUN: bash -- %s phy-translate | FileCheck %s
# CHECK: OVERVIEW: Phy translation tool

PHY_TRANSLATE=$1
${PHY_TRANSLATE} --help