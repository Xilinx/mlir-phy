//===- Resource.cpp ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "phy/Connectivity/Resource.h"

#include <iterator>
#include <list>
#include <sstream>
#include <string>

using namespace phy::connectivity;

Resource::Resource(std::string s) {
  size_t pos = 0;
  std::string key, value;

  while ((pos = s.find(delim)) != std::string::npos) {
    key = s.substr(0, pos);
    s.erase(0, pos + delim.length());
    if ((pos = s.find(delim)) != std::string::npos) {
      value = s.substr(0, pos);
      s.erase(0, pos + delim.length());
      metadata[key] = value;
    }
  }

  this->key = s;
}

std::string Resource::toString() {
  std::ostringstream result;
  for (auto info : metadata) {
    result << info.first << delim << info.second << delim;
  }
  result << key;

  return result.str();
}
