//===- Graph.cpp ------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "phy/Connectivity/Graph.h"

#include <iterator>
#include <list>
#include <sstream>
#include <string>

using namespace phy::connectivity;

Phy::Phy(std::string slash_seperated_info) {
  std::string s = slash_seperated_info;

  size_t pos = 0;
  std::string key, value;
  while ((pos = s.find(delim)) != std::string::npos) {
    key = s.substr(0, pos);
    s.erase(0, pos + delim.length());

    if ((pos = s.find(delim)) != std::string::npos) {
      value = s.substr(0, pos);
      s.erase(0, pos + delim.length());
      metadata[key] = value;

    } else {
      key = key;
    }
  }
}

std::string Phy::toString() {
  std::ostringstream result;
  for (auto info : metadata) {
    result << info.first << delim << info.second << delim;
  }
  result << key;

  return result.str();
}

Slot::Slot(std::string comma_seperated_phys) {
  std::string s = comma_seperated_phys;
  size_t pos = 0;
  std::string token;
  while ((pos = s.find(delim)) != std::string::npos) {
    token = s.substr(0, pos);
    s.erase(0, pos + delim.length());
    phys.push_back(Phy(token));
  }
  phys.push_back(Phy(s));
}

std::string Slot::toString() {
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
