# TTI FMA-fusion cost — status and fix scope

Follow-up to Alexey's review note on #210399:
*"need to fix it somehow to make TTI drop the assumption that fmul-fadd is folded to fma."*

Not needed for #210399 (already correct via the per-type query). This is the shape of the "real" fix, for a separate RFC/branch.

## The problem

`getArithmeticInstrCost` conflates two different questions for an FMUL:

1. What does this fmul cost as an operation? (standalone)
2. What does it cost *given it will fuse* into an fma? (contextual — ~0, because the fma is charged on the fadd)

It answers (2) when a context instruction is passed and (1) when not — **implicitly, as a side-effect of the `CxtI` argument**. A caller that holds the instruction but wants its standalone cost (e.g. because it is about to *break* the fusion) has no way to ask; it silently gets 0.

## Current status in the code

**The side-effect — AMDGPU only.** `llvm/lib/Target/AMDGPU/AMDGPUTargetTransformInfo.cpp:585`

```cpp
case ISD::FMUL:
  // ... return zero cost for fmul(b,c) supposing the fadd|fsub will get
  // estimated cost for the whole fused operation.
  if (CxtI && CxtI->hasOneUse())
    if (const auto *FAdd = dyn_cast<BinaryOperator>(*CxtI->user_begin())) {
      ... if (FAdd->hasAllowContract() && CxtI->hasAllowContract())
          return TargetTransformInfo::TCC_Free;   // fmul priced at 0
```

**The mechanism.** `getInstructionCost(I)` forwards `I` as `CxtI` to `getArithmeticInstrCost`, so per-instruction queries trigger the discount. A per-*type* query — `getArithmeticInstrCost(Opcode, Ty)`, `CxtI == nullptr` (`TargetTransformInfo.h:1544`) — never enters the branch and returns the full cost.

**Cross-target reach.** `CxtI` is a shared trailing parameter of `getArithmeticInstrCost`. 10 targets consult it, but **only AMDGPU** uses it for the fmul→fadd fusion discount. The rest read it for unrelated operand analysis. So the *semantic* footgun is AMDGPU-local; the *API* is shared.

**In-tree workaround.** SLP costs the fmul by type to dodge the discount:
- ordered-reduction penalty — `SLPVectorizer.cpp:30481` (`getArithmeticInstrCost(FMul, Ty, ...)`)
- `canConvertToFMA` internal fmul cost (on the canConvertToFMA branch, BUG2)

Correct, but the intent ("give me the un-fused cost") is expressed only by API choice, not stated.

## What has to be fixed

Make fusion-awareness **explicit** instead of inferred from `CxtI`.

- **Preferred — drop the implicit discount.** Have `getArithmeticInstrCost(FMUL)` always return the standalone op cost. Model the fma saving where the fusion is actually formed, via `getIntrinsicInstrCost(fmuladd)`. Then:
  - delete AMDGPU's `FMUL`/`CxtI` branch;
  - audit callers that relied on the discount (LoopVectorize, SLP, unroll/inline cost) and have them charge fusion explicitly;
  - SLP's per-type calls become plain standalone queries — intent now obvious.
- **Alternative — explicit flag.** Add `bool AssumesFMAFusion` (default false → standalone) to `getArithmeticInstrCost`. Less caller churn, but grows an already-long signature and keeps two cost modes.

## Scope / risk

- Touches a shared, heavily-overridden TTI method → cross-target, **RFC required**.
- Changes AMDGPU cost outputs for every current consumer of the discount → needs measurement before/after.
- Zero correctness impact on #210399; this is a modeling cleanup, not a bug fix.
