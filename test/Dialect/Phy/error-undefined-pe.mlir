// RUN: (phy-opt %s 2>&1 || true) | FileCheck %s

// CHECK-LABEL: 'phy.pe' op expected symbol reference func to point to a function
%pe = phy.pe @func()