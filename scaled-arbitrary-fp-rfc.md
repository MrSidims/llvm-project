# RFC: Scaled Arbitrary Floating-Point Conversion Intrinsics

## Summary

Propose new LLVM IR intrinsics `llvm.convert.to.scaled.arbitrary.fp` and
`llvm.convert.from.scaled.arbitrary.fp` that extend the existing arbitrary
FP conversion intrinsics with an explicit scale factor, enabling hardware
acceleration of MX (Microscaling) format quantization on AMDGPU and
potentially other targets.

## Motivation

The OCP Microscaling Formats specification defines block-scaled quantization
where groups of narrow FP values (FP4/FP6/FP8) share a common scale factor
encoded as Float8E8M0FNU (8-bit exponent-only format, bias=127). This is the
dominant quantization scheme in ML inference (used by Gluon, Triton, PyTorch).

AMDGPU hardware (gfx950, gfx1250) has native instructions that fuse the
scale multiplication with format conversion:

| Instruction | Operation | GPU |
|---|---|---|
| `v_cvt_scalef32_pk_f32_fp8` | `v2f32 = unpack(i32_fp8) * scale` | gfx950, gfx1250 |
| `v_cvt_scalef32_pk_fp8_f32` | `v2i16_fp8 = pack(f32/scale)` | gfx950, gfx1250 |
| `v_cvt_scalef32_pk_f32_fp4` | `v2f32 = unpack(i32_fp4) * scale` | gfx950, gfx1250 |
| `v_cvt_scalef32_pk32_f32_fp6` | `v32f32 = unpack(v6i32_fp6) * scale` | gfx950, gfx1250 |

The existing `llvm.convert.from/to.arbitrary.fp` intrinsics have no scale
parameter, so these fused instructions cannot be used.

## Proposed Intrinsics

### `llvm.convert.from.scaled.arbitrary.fp`

```llvm
declare <ResultTy> @llvm.convert.from.scaled.arbitrary.fp.<ResultTy>.<SrcIntTy>(
    <SrcIntTy> %bits,     ; integer containing arbitrary FP bits
    metadata %format,      ; format string (e.g., "Float8E4M3FN")
    float %scale           ; block scale factor (applied after conversion)
) nounwind readnone speculatable
```

Semantics: `result = convert(bits, format) * scale`

### `llvm.convert.to.scaled.arbitrary.fp`

```llvm
declare <ResultIntTy> @llvm.convert.to.scaled.arbitrary.fp.<ResultIntTy>.<SrcFloatTy>(
    <SrcFloatTy> %value,  ; native float value
    metadata %format,      ; target format string
    metadata %round,       ; rounding mode
    i1 %saturate,          ; saturation flag
    float %scale           ; block scale factor (applied before conversion)
) nounwind readnone speculatable
```

Semantics: `result = convert(value / scale, format, round, saturate)`

### Relationship to existing intrinsics

When `scale = 1.0`, the scaled intrinsics are equivalent to the unscaled
versions. Targets that don't have fused scale instructions can expand them
as:

```
from_scaled(bits, fmt, scale) → fmul(convert_from(bits, fmt), scale)
to_scaled(val, fmt, round, sat, scale) → convert_to(fdiv(val, scale), fmt, round, sat)
```

## Bridge Strategy: DAG Combine

Before the new intrinsics are adopted, a DAG combine approach can fuse
existing IR patterns into scaled HW instructions:

### Pattern 1: Scaled FROM conversion
```llvm
%raw = call float @llvm.convert.from.arbitrary.fp.f32.i8(i8 %bits, metadata !"Float8E4M3FN")
%result = fmul float %raw, %scale
```
DAG combine: `fmul(CONVERT_FROM_ARBITRARY_FP(src, sem), scale)` →
`INTRINSIC(amdgcn_cvt_scalef32_f32_fp8, src, scale, sel)`

### Pattern 2: Scaled TO conversion
```llvm
%scaled = fdiv float %val, %scale    ; or fmul with 1/scale
%r = call i8 @llvm.convert.to.arbitrary.fp.i8.f32(float %scaled, ...)
```
DAG combine: `CONVERT_TO_ARBITRARY_FP(fdiv(val, scale), sem, ...)` →
`INTRINSIC(amdgcn_cvt_scalef32_pk_fp8_f32, ..., scale, ...)`

### Precision consideration

The fused HW instruction may produce slightly different results than the
separate fmul + convert sequence due to intermediate rounding. The DAG
combine should be gated on `fast-math` flags or an explicit opt-in.

## FP6 Note

FP6/BF6 HW instructions on AMDGPU operate on 32 values at a time
(`v_cvt_scalef32_pk32_f32_fp6`). Efficient lowering requires vector-width
matching (32 elements). Scalar or small-vector FP6 conversions should
continue to use generic expansion unless the vector width matches.

## Target Lowering

### AMDGPU
- `hasFP8ConversionScaleInsts()`: FP8/BF8 scaled → `v_cvt_scalef32_*_fp8/bf8`
- `hasFP4ConversionScaleInsts()`: FP4 scaled → `v_cvt_scalef32_*_fp4`
- `hasFP6BF6ConversionScaleInsts()`: FP6/BF6 scaled → `v_cvt_scalef32_pk32_*`
- When scale=1.0: unscaled conversion using scale instructions (already implemented for FP4)

### Default expansion
Targets without HW support: `fmul/fdiv + convert_from/to_arbitrary_fp`

---

## Semantic Precision

This section pins down the corner-case semantics of the
`llvm.convert.{from,to}.arbitrary.fp[.scaled]` family so that round-trip
equivalence (e.g., between SW expansion and HW lowering, or between two
different targets) is well defined. Without these contracts, "bit-exact"
matches between paths cannot be asserted.

### Saturation semantics

The `i1 %saturate` flag on `llvm.convert.to.arbitrary.fp` controls overflow
behavior. The result depends on the destination format's
`fltNonfiniteBehavior`:

| Destination `fltNonfiniteBehavior` | `saturate=true` | `saturate=false` |
|---|---|---|
| `IEEE754` (E5M2, future E4M3) | overflow → ±max-finite | overflow → ±Inf |
| `NanOnly` (E4M3FN, FNUZ family) | overflow → ±max-finite | overflow → ±max-finite (no Inf to produce) |
| `FiniteOnly` (FP4, FP6) | overflow → ±max-finite, NaN input → poison | overflow → ±max-finite, NaN input → poison |

For `+Inf` / `-Inf` source on a no-Inf destination format the behavior
follows the same overflow rule.

### NaN payload handling

- Source NaN → destination canonical qNaN. Payload bits are not preserved.
- For `IEEE754` destinations: NaN encoding is `sign=0, exp=all-ones,
  qNaN-bit=1, other mantissa bits=0`.
- For `NanOnly` destinations with `nanEncoding == AllOnes` (E4M3FN): NaN
  encoding is `exp=all-ones, mant=all-ones`.
- For `NanOnly` destinations with `nanEncoding == NegativeZero` (the FNUZ
  family): NaN encoding is `sign=1, exp=0, mant=0`.
- For `FiniteOnly` destinations (FP4, FP6, E8M0FNU): a NaN input produces
  poison. (E8M0FNU is a special case: it has a single NaN encoding at
  255, so NaN inputs produce that encoding rather than poison.)

### Denormal handling

The intrinsic respects the surrounding function's `denormal-fp-math`
attribute:

- `preserve-sign,preserve-sign`: input denormals are flushed to ±0 before
  conversion; the destination format never produces denormals.
- `ieee,ieee`: full IEEE denormal handling on both sides.
- `dynamic`: target backend chooses based on the dynamic FP environment.
  AMDGPU uses `MODE.fp32_denorm`.

The expansion in `LegalizeDAG.cpp` and the AMDGPU custom lowering both
follow this contract by default; targets that flush denormals at the HW
level will silently apply that flush regardless of the IR attribute.

### Underflow

When the magnitude of the source value falls below the smallest subnormal
representable in the destination format:

- Round-to-nearest-even (RNE) and round-toward-zero (RTZ): produce ±0.
- Directed rounding (round-toward-positive / round-toward-negative): round
  to the nearest representable value in the rounding direction, which may
  be the smallest subnormal or ±0 depending on sign.

### Stochastic rounding intrinsic shape (design only)

Stochastic rounding requires an explicit entropy source, which the existing
`metadata %round` operand cannot carry. We propose a *separate* intrinsic
family rather than adding an optional operand to the existing one:

```llvm
declare <IntTy> @llvm.convert.to.arbitrary.fp.sr.<IntTy>.<FloatTy>(
    <FloatTy> %value, metadata %format, i32 %seed, i1 %saturate)
```

Semantics:

```
result = round_to_nearest(value + seed_perturbation, format)
```

where `seed_perturbation = (seed mod 2^N) / 2^N * ulp(value)` and `N` is
the number of mantissa bits truncated by the conversion.

Properties:
- The `i32 %seed` is required. There is no implicit RNG and no global
  state. The caller is responsible for the entropy source (typically a
  per-lane PRNG or counter).
- Two calls with the same `(value, seed)` are deterministic and produce
  bit-identical results.
- HW lowering on AMDGPU maps to the `v_cvt_sr_*` family with the seed
  passed through as the `i32 seed` operand.
- Generic expansion applies the seed-derived perturbation to the source
  mantissa before performing round-to-nearest.

This design is fixed here. Implementation is a follow-up.

### Stochastic + saturate ordering

When both stochastic rounding and saturation are in effect, the
conceptual order is:

1. Perturb `value` by the seed-derived ULP fraction.
2. Round to nearest in the destination format.
3. Saturate if the result exceeds the destination range.

This matches the AMDGPU `v_cvt_sr_*` HW behavior and avoids ambiguity at
the saturation boundary.

### Vector input / packed integer output design

The intrinsic accepts vector floating-point inputs as first-class
operands across the natural pack widths:

- `<2 x f32>`, `<4 x f32>`, `<8 x f32>`, `<16 x f32>`, `<32 x f32>`
- The corresponding f16 / bf16 variants

Outputs are *packed integer types* matching HW lane width, not narrow
vector types:

| Input width | FP8 output | FP4 output | FP6 output |
|---|---|---|---|
| `<2 x f32>` | `i16` | `i8` | `i16` (12 bits used) |
| `<4 x f32>` | `i32` | `i16` | `i32` |
| `<8 x f32>` | `i64` | `i32` | `i64` (48 bits used) |
| `<16 x f32>` | `<2 x i64>` | `i64` | `<3 x i32>` |
| `<32 x f32>` | `<4 x i64>` | `<2 x i64>` | `<6 x i32>` |

Rationale: this sidesteps the type legalizer's
`getPreferredVectorAction → TypeSplitVector` action on `<N x i8>` /
`<N x i4>` (which would otherwise scalarize before custom lowering can
intervene). It also matches HW pack widths exactly: `pk32_fp6` returns
`<6 x i32>`, `pk_fp8_f32` returns `v2i16` (one i32), etc.

Frontends bitcast on either side. The vector lowering rewrite is a
follow-up implementation; the API contract is fixed here.

### Format string canonicalization

- Format strings are case-sensitive APFloat semantics names:
  `"Float8E4M3FN"`, `"Float8E5M2"`, `"Float8E8M0FNU"`, `"Float8E5M2FNUZ"`,
  `"Float8E4M3FNUZ"`, `"Float8E4M3B11FNUZ"`, `"Float6E2M3FN"`,
  `"Float6E3M2FN"`, `"Float4E2M1FN"`.
- The verifier rejects unrecognized strings via
  `APFloatBase::isValidArbitraryFPFormat`.
- Aliases are *not* supported. Frontends must emit the canonical APFloat
  name (no `"FP8E4M3"` / `"E4M3"` / `"e4m3fn"` shortcuts).
