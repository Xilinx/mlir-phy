//===- Implementation.cpp ---------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "phy/Connectivity/Implementation.h"

#include <iterator>
#include <list>
#include <sstream>
#include <string>

using namespace phy::connectivity;

Implementation::Implementation(std::string s) {
  size_t pos = 0;
  std::string token;
  while ((pos = s.find(delim)) != std::string::npos) {
    token = s.substr(0, pos);
    s.erase(0, pos + delim.length());
    phys.push_back(PhysicalResource(token));
  }
  phys.push_back(PhysicalResource(s));
}

std::string Implementation::toString() {
  std::ostringstream result;
  bool first = true;

  for (auto phy : phys) {
    if (!first) {
      result << delim;
    }
    result << phy.toString();
    first = false;
  }

  return result.str();
}
