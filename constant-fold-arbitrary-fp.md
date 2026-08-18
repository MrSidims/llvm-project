# Constant folding for llvm.convert.{to,from}.arbitrary.fp

## What was implemented

IR-level constant folding for the two intrinsics, reachable from InstSimplify /
InstCombine / SCCP / GVN etc. via `ConstantFoldCall`.

Files:
- `llvm/lib/Analysis/ConstantFolding.cpp`
  - `canConstantFoldCallTo` now returns true for both intrinsics.
  - `getArbitraryFPFormat` reads the interpretation metadata off the call.
  - `ConstantFoldConvertToArbitraryFP` / `ConstantFoldConvertFromArbitraryFP`
    implement the scalar conversions with APFloat.
  - `ConstantFoldConvertArbitraryFP` dispatches and is called early in
    `ConstantFoldScalarCall` (metadata operands are stripped from `Operands`
    by `tryConstantFoldCall`, so they are read from the `Call`).
- `llvm/lib/Analysis/VectorUtils.cpp` — `isVectorIntrinsicWithScalarOpAtArg`
  marks the metadata and saturation operands scalar so the per-lane vector
  folding path works.
- `llvm/test/Transforms/InstSimplify/ConstProp/convert-arbitrary-fp.ll`

The fold matches `TargetLowering::expand{To,From}CONVERT_ARBITRARY_FP`:
- NaN -> canonical positive qNaN, or poison for finite-only formats.
- Inf -> signed inf, else saturate-to-max, else poison.
- Zero -> signed zero.
- Finite overflow -> saturate-to-max, else signed inf, else poison.
- Overflow detection rounds into a probe format (destination precision + one
  extra exponent bit + IEEE non-finite behavior) so formats that clamp on
  overflow do not hide the overflow.

### Validation

Exhaustive cross-check against the runtime expansion (compile the unfolded IR
with llc, execute, compare to the folded constants): 4412 scalar cases across
all five formats, five rounding modes, both saturation values, and every
special value, plus width-mismatch and bf16/vector spot checks.
- 3951 defined results match exactly.
- 455 poison results are all legitimate (verified: NaN->finite-only,
  Inf->no-inf-no-sat, finite-overflow->no-inf-no-sat). No false poison.
- 6 mismatches are a PRE-EXISTING codegen bug, not a fold bug (see B below).

An independent review agent re-ran ~800k differential comparisons (opt fold vs
llc expansion, with ml_dtypes as a third oracle) and confirmed: correct
everywhere except the same pre-existing expansion bug; no assertion / UB /
lifetime issue with the probe fltSemantics; metadata reading, vector scalar-op
indices, width handling, special values, overflow probe, and format gating all
correct.

## Discovered pre-existing bug

`expandCONVERT_FROM_ARBITRARY_FP` mishandles a source denormal that is still
denormal in the destination. Example: Float8E5M2 `0x01` = 2^-16 converted to
`half` should be `0x0100`, the runtime expansion produces `0xFC00` (-inf). The
float path is fine because 2^-16 is normal in f32. Root cause: the denormal
conversion path (CTLZ based) assumes the result is normal in the destination;
when the destination is also denormal the computed exponent underflows. The new
constant folder produces the correct value.

## Planned follow-up optimizations (not implemented)

### A. SelectionDAG / GlobalISel constant folding  [high]
llc does not fold the expansion for constant inputs today, because `ISD::FFREXP`
of a constant is not folded by `getNode`, so the whole bit-manipulation stays as
runtime code. Two options, best combined:
1. Fold `ISD::FFREXP` of a constant in `getNode` / DAGCombiner. General, helps
   every FFREXP user, and makes the existing expansion collapse for constants.
2. Fold `ISD::CONVERT_{TO,FROM}_ARBITRARY_FP` with constant operands directly
   (in `SelectionDAGBuilder` before emitting the node, or in DAGCombiner).
To avoid a second copy of the logic, extract the scalar conversion into ONE
shared routine both the IR folder and codegen call. Poison is not an APFloat
concept, so the shared routine returns a value plus an is-poison flag; a natural
home is `APFloat` (`convertToArbitraryFP` / `convertFromArbitraryFP`) with the
poison decision left to the caller.

### B. Fix the denormal->denormal expansion bug  [correctness]
Fix `expandCONVERT_FROM_ARBITRARY_FP` to handle the case where the widened
result is still denormal in the destination. Add exhaustive denormal->half
tests. Afterwards the folder and codegen agree on those inputs.

### C. InstCombine round-trip peephole  [medium]
`to(from(y, fmt), fmt, rm, sat) -> y`. `from` yields an exactly representable
value, so `to` of it is exact and rm/sat are irrelevant. Valid for finite / inf
/ zero. Guard NaN: formats with NaN canonicalize a non-canonical payload, so
only apply when `y` is not a non-canonical NaN (always safe for finite-only
formats), and require the integer width to equal the format width. Corollary:
idempotent quantization `to(from(to(x))) -> to(x)`.

### D. computeKnownFPClass / computeKnownBits  [medium, partly in-flight]
- `from(y, fmt)` for finite-only formats is never NaN and never Inf; sign and
  zero classes derive from the sign bit. (Overlaps PR #208585.)
- `to(x, fmt)` clears all result bits at or above the format width; teach
  computeKnownBits so downstream and/zext/trunc simplify, especially when the
  result integer type is wider than the format.

### E. Poison / undef propagation  [low]
Add both intrinsics to `intrinsicPropagatesPoison` after confirming callers rely
only on the forward direction (these can produce poison from non-poison inputs).
Fold `to(undef)` / `from(undef)`.

### F. Scalable-vector constant folding  [low]
`ConstantFoldScalableVectorCall` does not fold these; a splat-constant scalable
vector could fold element-wise.

### G. Vectorization support  [low, larger]
Teach `isTriviallyVectorizable` / VFABI so LoopVectorize / SLP can vectorize the
scalar calls. Complicated by the metadata operands.

### H. from + fpext / fptrunc combines  [low]
`fpext(from(y, fmt))` equals `from(y, fmt)` straight to the wider type because
`from` is an exact widening; fold into a single wider `from`.
