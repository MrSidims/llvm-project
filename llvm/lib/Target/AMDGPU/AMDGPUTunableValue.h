//===- AMDGPUTunableValue.h - Flag and attribute knobs ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Resolves a heuristic constant that can be set either per module through a
/// command line option or per function through an attribute of the same name.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUTUNABLEVALUE_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUTUNABLEVALUE_H

#include "llvm/IR/Function.h"
#include "llvm/Support/CommandLine.h"

namespace llvm {

/// Returns the value of \p Opt for function \p F. An explicitly given command
/// line option always wins, otherwise a function attribute named after the
/// option supplies the value, and failing that the option default is used.
inline unsigned getTunableValue(const Function &F,
                                const cl::opt<unsigned> &Opt) {
  if (Opt.getNumOccurrences())
    return Opt;
  return F.getFnAttributeAsParsedInteger(Opt.ArgStr, Opt);
}

inline bool getTunableValue(const Function &F, const cl::opt<bool> &Opt) {
  if (Opt.getNumOccurrences())
    return Opt;
  Attribute Attr = F.getFnAttribute(Opt.ArgStr);
  return Attr.isStringAttribute() ? Attr.getValueAsBool() : Opt;
}

} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUTUNABLEVALUE_H
