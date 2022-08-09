// RUN: (phy-opt %s 2>&1 || true) | FileCheck %s

// CHECK-LABEL: 'physical.core' op expected symbol reference func to point to a function
%pe = physical.core @func() : () -> !physical.core