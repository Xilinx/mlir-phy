//===- ResourceList.cpp -----------------------------------------*- C++ -*-===//
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "phy/Connectivity/ResourceList.h"

#include <iterator>
#include <list>
#include <sstream>
#include <string>

using namespace phy::connectivity;

ResourceList::ResourceList(std::string s) {
  size_t pos = 0;
  std::string token;
  while ((pos = s.find(delim)) != std::string::npos) {
    token = s.substr(0, pos);
    s.erase(0, pos + delim.length());
    phys.push_back(PhysicalResource(token));
  }
  phys.push_back(PhysicalResource(s));
}

std::string ResourceList::toString() {
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
