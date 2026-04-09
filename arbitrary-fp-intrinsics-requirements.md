# Requirements: LLVM Arbitrary-FP Conversion Intrinsics for Triton/Gluon

## Goal

Replace Triton's hand-rolled FP8/FP6/FP4 conversion code (`ElementwiseOpToLLVM.cpp`,
~600 lines of `_HW`/`_SW` dispatch + bit manipulation) with a single set of LLVM IR
intrinsics, then have *the LLVM backend* assume responsibility for choosing between
hardware instructions and software emulation per target. The same IR must round-trip
through SPIR-V (both `spirv64` Khronos and `spirv64-amd-amdhsa` vendor flows) without
semantic loss.

The target architecture is:

```
                  Triton/Gluon dialect (FpToFpOp, MX quantize)
                                  │
                                  ▼
              llvm.convert.{to,from}[.scaled].arbitrary.fp(...)
                                  │
              ┌───────────────────┼────────────────────┐
              ▼                   ▼                    ▼
      AMDGPU backend         SPIR-V backend       Other targets
              │                   │                    │
              ▼                   ▼                    ▼
   HW (per ISA) or       Preserve as ext call     Generic LegalizeDAG
   generic expansion     or expand to portable    expansion (already
                         SPIR-V                    present)
                                  │
                                  ▼
                         SPIR-V translator
                                  │
                                  ▼
                   Reconstruct original intrinsic
                                  │
                                  ▼
                         AMDGPU backend (HW)
```

The deliverable is *correctness equal to Triton's current `_SW` path on all ISAs* and
*performance equal to or better than Triton's current `_HW` path on ISAs where HW exists*.

---

## R1. Intrinsic API

### R1.1 — Coverage of formats

The intrinsic `format` metadata string must accept **every APFloat semantics** Triton can
produce, not just the OCP MX subset currently in the diff:

| Family | Formats | Current status | Required |
|---|---|---|---|
| OCP MX | E5M2, E4M3FN, E3M2FN, E2M3FN, E2M1FN | Implemented (expansion) | Keep |
| OCP MX scale | **E8M0FNU** | TODO in expansion | **Required** — without it, MX block quantize/dequantize cannot lower |
| FNUZ (CDNA3) | E5M2FNUZ, E4M3FNUZ, E4M3B11FNUZ | TODO in expansion | **Required** — Triton currently has CDNA3-specific HW paths for these (`Fp32_to_Fp8E5M2FNUZ_HW`); the intrinsic must subsume them |
| Other IEEE-style | E4M3 (with Inf), E3M4 | TODO | Lower priority — Triton doesn't currently emit these |

The intrinsic must reject (with verifier error, not crash) any format string that does
not resolve via `APFloatBase::getArbitraryFPSemantics`.

### R1.2 — Rounding modes

The `rounding` metadata operand must accept at minimum:

- `round.tonearest` (RTNE) — Triton's default
- `round.towardzero` (RTZ) — Triton uses this for `Fp16_to_Fp8E5M2_RTZ`
- `round.tonearestaway` — for completeness
- `round.upward`, `round.downward` — for completeness
- **`round.stochastic`** — *new*, required for ML training paths. Maps to `v_cvt_sr_*`
  on gfx942+. The current `convertStrToRoundingMode` does not parse this; either reuse
  `RoundingMode::NearestTiesToAway` overloaded as stochastic (bad — different semantics)
  or extend `FloatingPointMode.h` with a `Stochastic` enum value (better).

The intrinsic must accept stochastic rounding only when the call site provides a
seed/random source operand. The simplest design: add an optional `i32` operand for the
random seed; if absent, fall back to RTNE.

### R1.3 — Scale operand (the RFC)

The unscaled intrinsics in the current diff cannot reach hardware throughput on
gfx950/gfx1250 because the dominant pattern for MX quantization is
`convert(value / scale, format)` where `scale` is a per-block E8M0FNU exponent. The fused
HW instructions (`v_cvt_scalef32_pk_fp8_f32`, `v_cvt_scalef32_pk16_fp6_f32`,
`v_cvt_scalef32_pk32_*`) take this scale as an operand.

**Required:** add scaled variants per `scaled-arbitrary-fp-rfc.md`:

```
declare <ResultIntTy> @llvm.convert.to.scaled.arbitrary.fp.<ResultIntTy>.<SrcFloatTy>(
    <SrcFloatTy> %value, metadata %format, metadata %round,
    i1 %saturate, float %scale)

declare <ResultFloatTy> @llvm.convert.from.scaled.arbitrary.fp.<ResultFloatTy>.<SrcIntTy>(
    <SrcIntTy> %bits, metadata %format, float %scale)
```

When `scale == 1.0` constant, these must be canonicalized to the unscaled form so the
existing FP4 scale=1.0 trick keeps working without duplicating lowering paths. Targets
without fused scale instructions expand to `fmul/fdiv + convert_from/to_arbitrary_fp`
(already specified in the RFC).

### R1.4 — Vector types as first-class operands

The current vector handling scalarizes via `ScalarizeVecRes_*` then re-merges via DAG
combine. This is fragile and produces suboptimal code for the common Triton case
(`<8 x f32> → <8 x i8>`).

**Required:** the intrinsic must accept arbitrary vector widths (`<2 x f32>`,
`<4 x f32>`, `<8 x f32>`, `<16 x f32>`, `<32 x f32>`) and the AMDGPU lowering must emit
packed instructions directly without going through scalarize → combine. Specifically,
vector widths matching native packed instruction widths must produce single instructions:

- `<2 x f32>` → `v_cvt_pk_fp8_f32` (1 inst)
- `<8 x f32>` → `v_cvt_scalef32_pk8_fp8_f32` on gfx950+ (1 inst)
- `<16 x f32>` → `v_cvt_scalef32_pk16_fp6_f32` on gfx950+ (1 inst)
- `<32 x f32>` → `v_cvt_scalef32_pk32_fp6_f32` on gfx950+ (1 inst)
- Larger vectors: split, not scalarize

### R1.5 — Result/source type matrix

| Source type | Required result types (CONVERT_TO) |
|---|---|
| f16, bf16 | i4, i6, i8 (scalar + vector) |
| f32 | i4, i6, i8 (scalar + vector) |
| f64 | i8 (scalar only — no HW path, generic only) |

| Source type | Required result types (CONVERT_FROM) |
|---|---|
| i4, i6, i8 | f16, bf16, f32 (scalar + vector) |

Triton currently has explicit `Fp8 → Bf16`, `Fp16 → Bf16` paths that the intrinsic must
cover. The current diff only handles `f16` and `f32` results — `bf16` is missing and is
a real Triton path.

---

## R2. AMDGPU Backend Lowering

The AMDGPU custom lowering must form a complete dispatch matrix. The current diff covers
~5 cells of the matrix; the full requirement is ~40 cells. Below is the target table;
"Generic" means fall through to `LegalizeDAG.cpp` expansion.

### R2.1 — Per-architecture instruction availability

| Arch | OCP FP8 unscaled | OCP FP8 scaled | OCP FP4/FP6 scaled | FNUZ | SR rounding | f16 direct |
|---|---|---|---|---|---|---|
| gfx906/908/90a/RDNA1-3 | Generic | Generic | Generic | Generic | Generic | Generic |
| gfx942 (CDNA3) | `v_cvt_pk_fp8_f32` | Generic | Generic | `v_cvt_pk_fp8_f32` (FNUZ semantics) | `v_cvt_sr_fp8_f32` | `v_cvt_pk_fp8_f16` |
| gfx950 (CDNA4) | via scaled w/ scale=1.0 | `v_cvt_scalef32_pk_fp8_f32`, `pk8_fp8_*` | `pk8_fp4_*`, `pk16_fp6_*`, `pk32_fp6_*` | Generic (CDNA3-only HW) | `v_cvt_scalef32_sr_*` | `pk_fp8_f16`, `pk8_fp8_f16` |
| gfx1100-1153 (RDNA3) | Generic | Generic | Generic | Generic | Generic | Generic |
| gfx1200/1201 | `v_cvt_pk_fp8_f32` | mul + unscaled | Generic | Generic | yes | yes |
| gfx1250 | Same as gfx950 + wider | Same as gfx950 + `pk8`/`pk16` direct | Same as gfx950 + `pk8`/`pk16`/`pk32` | Generic | Same as gfx950 | Native via direct intrinsics |

Critical observation: **gfx942 currently has zero HW acceleration in Triton's path**
(Triton's `_HW` is gated on `isCDNA4OrHigher`). The new backend lowering must use
`v_cvt_pk_fp8_f32` on gfx942 — this is a strict win over Triton's current behavior on
that ISA.

### R2.2 — Subtarget feature gates

The lowering must use the existing fine-grained subtarget features rather than coarse
`isCDNA4OrHigher`-style checks:

- `hasFP8ConversionInsts()` — gfx940/942/1170/1200/1250 — basic `v_cvt_pk_fp8_f32`
- `hasFP8ConversionScaleInsts()` — gfx950, gfx1250 — `v_cvt_scalef32_*_fp8`
- `hasFP4ConversionScaleInsts()` — gfx950, gfx1250 — `v_cvt_scalef32_*_fp4`
- `hasFP6BF6ConversionScaleInsts()` — gfx950, gfx1250 — `v_cvt_scalef32_pk16/pk32_*_fp6`
- `hasFP8E5M3Insts()` — gfx1250-specific E5M3 variant
- `hasGFX1250Insts()` — gfx1250 only — direct f16↔fp8 paths

### R2.3 — Lowering rules

For each `(format, src_type, dst_type, rounding, saturate, scale, target)` tuple, the
lowering must:

1. Try the most specific HW instruction first (e.g., `pk32_fp6_f32` for
   `<32 x f32> → <32 x i6>` on gfx950).
2. If no exact match, try a wider HW instruction with masking (e.g., scalar
   `f32 → i8` via `pk_fp8_f32` with undef second slot, mask byte 1 — already done in
   current diff for FP8).
3. If no scaled HW instruction but a scaled instruction exists, emit it with
   `scale=1.0` (already done for FP4).
4. If no HW instruction at all, return `SDValue()` and fall through to generic
   expansion.

The lowering must **never silently degrade**: if a target has the HW instruction but the
rounding mode/saturation flag/source type combination isn't supported by HW, falling
through to generic expansion is acceptable, but a target-feature-aware test must verify
this happens (no missed optimizations).

### R2.4 — DAG combine for fused scale

Until frontends adopt the scaled intrinsics, a DAG combine must fuse:

- `fmul(CONVERT_FROM_ARBITRARY_FP(bits, fmt), scale)` → scaled HW instruction
- `CONVERT_TO_ARBITRARY_FP(fdiv(val, scale), fmt, ...)` → scaled HW instruction
- `CONVERT_TO_ARBITRARY_FP(fmul(val, recip), fmt, ...)` → scaled HW instruction (if
  `recip` is `1/scale`)

This combine must be gated on `nsz nnan ninf` fast-math flags or an explicit opt-in,
because the fused HW instruction may differ from the unfused sequence by one ULP at the
rounding boundary.

### R2.5 — Composition with MFMA/WMMA

The lowering must not insert spurious convert-and-back-again sequences when a
CONVERT_FROM result feeds an MFMA that takes raw FP8 bits, or when a CONVERT_TO result
is consumed by an MFMA. In particular:

- Pattern `MFMA_F8F8(CONVERT_FROM(bits))` must collapse to `MFMA_F8F8(bits)` directly.
- Pattern `CONVERT_TO(MFMA_F32F32_result)` followed by store must use the packed
  `pk_fp8_f32` form, not scalar-then-merge.

### R2.6 — `performOrCombine` extension

The current packed-pair merge in `performOrCombine` matches only the f32 and f16 PK FP8
patterns. It must be extended to:

- FP4 nibble pairs (4 conversions packed into one byte via `pk8_fp4_f32`)
- FP6 (6-bit packing is non-byte-aligned; this is harder, may require a different combine)
- The scaled variants (currently only matches the unscaled `pk_fp8_f32`)

Alternatively (better long term): emit packed instructions directly from the vector
lowering path (R1.4) so the combine isn't needed at all.

---

## R3. Generic Expansion (LegalizeDAG)

### R3.1 — Format coverage

The current expansion handles 5 OCP formats. It must additionally handle:

- **Float8E8M0FNU** — exponent-only scale type. Special case: no mantissa, no sign, no
  Inf, no denorm; conversion is essentially `extract exponent + bias adjust + saturate`.
  ~10 nodes vs ~80 for general formats.
- **FNUZ formats** — used by Triton's CDNA3 HW path today. Generic expansion is needed
  for non-CDNA3 targets and for the SPIR-V flow. The differences from FN: bias offset is
  +1, no negative zero (only one zero), single NaN encoding (sign=1, all-zeros).
- **Float8E4M3** (IEEE-style with Inf) — for completeness with non-AMD targets.

### R3.2 — Lookup-table fast path for tiny formats

For Float4E2M1FN (16 values) and Float6E*FN (64 values), the expansion can use a lookup
table (`select` chain or constant-pool load) instead of bit manipulation. This produces
~5-10 nodes instead of ~80 and is easier for the optimizer to constant-fold.

### R3.3 — Fast-math elision

When the source value is known finite (`nnan ninf` flags on the operand or `assume`
directive), the expansion must elide the NaN/Inf branches, cutting the node count
roughly in half. The current expansion always generates the full classification
regardless of flags.

### R3.4 — Vector expansion

The expansion must handle vector inputs without first scalarizing. Bit manipulation maps
trivially to vector ops; the only complication is the `getSetCC`/`select` shapes which
already work on vectors. This is critical for non-AMD targets where the vector code is
the only path.

---

## R4. Triton Frontend Integration

### R4.1 — Replace the dispatch table

`ElementwiseOpToLLVM.cpp` lines 1873-1917 (the `srcMap` lookup table) must be reduced to
a single entry per `(src_ty, dst_ty, rounding)` tuple that emits the new intrinsic, with
no `isaFamily` argument. The `_HW`/`_SW` function pairs and the `isCDNA4OrHigher`-gated
`ConverterT` factories (lines 501, 731, 852, 948, 1108, 1154, 1445, 1538, 1566, 1590,
etc.) must be deleted.

The architecture-aware code stays in *one place*: the AMDGPU backend.

### R4.2 — Scale handling at the frontend

When Triton lowers an MX block quantization (where it has access to the per-block scale
tensor), it must emit the scaled intrinsic, not unscaled + fmul. Otherwise the DAG
combine of R2.4 has to recover the pattern, which is fragile.

This requires extending `FpToFpOpConversion` to detect the pattern
`mx_quantize(value, scale)` at the Triton dialect level and emit
`llvm.convert.to.scaled.arbitrary.fp` directly.

### R4.3 — Drop the `cvtScalePk*UpcastFromFp8` / `cvtScalePk*DowncastToFp8` helpers

The helpers in `ElementwiseOpToLLVM.cpp:236-380` that emit `ROCDL::CvtScaleF32PkF32Fp8Op`
etc. must be deleted. Their job moves entirely into the AMDGPU backend's
`LowerCONVERT_*_ARBITRARY_FP`.

### R4.4 — Tests

Existing Triton FP8 conversion lit tests must pass unchanged. New numerical equivalence
tests must compare:

- Old SW path output ↔ new generic expansion output (bit-exact)
- Old HW path output ↔ new AMDGPU custom lowering output (bit-exact)
- Round-trip: `f32 → fp8 → f32` matches reference implementation

---

## R5. SPIR-V Backend (LLVM IR → SPIR-V)

### R5.1 — Two emission modes

The SPIR-V backend must distinguish:

- **Khronos mode** (`spirv64-unknown-unknown`, `spirv64-khronos-*`): emit standardized
  SPIR-V where possible. If no standard exists, fail compilation with a clear error
  rather than silently emitting target-specific code.
- **AMD vendor mode** (`spirv64-amd-amdhsa`): preserve the intrinsic as an external
  function call `spirv.llvm_convert_to_arbitrary_fp.<mangled>` with parameters carried
  through as decorations or aux-data.

### R5.2 — Khronos mode lowering

For Khronos mode, the backend must either:

1. **Use an existing extension** if one matches. Currently no Khronos SPIR-V extension
   covers OCP MX formats. The closest are `SPV_KHR_bfloat16` (bf16 only) and
   `SPV_INTEL_bfloat16_conversion`. For FP8, `SPV_INTEL_float8` is in draft. None
   cover FP4/FP6.
2. **Expand to portable SPIR-V** using the same algorithm as `LegalizeDAG` expansion,
   emitted as SPIR-V integer ops. This is the safe fallback and must be implemented.
3. **Propose a new Khronos extension** `SPV_KHR_arbitrary_fp_conversion` covering the
   OCP MX subset. This is a parallel-track effort, not a blocker.

### R5.3 — AMD vendor mode preservation

For `spirv64-amd-amdhsa`, the intrinsic must be preserved as an external call following
the same convention as the existing `spirv.llvm_amdgcn_*` shims. The format string,
rounding mode, and saturate flag must round-trip via:

- Encoding format string as a literal string operand on the SPIR-V function call (using
  `OpString` + `OpExtInst NonSemantic`)
- Encoding rounding mode and saturate flag as integer literal constants
- Using `--spirv-preserve-auxdata` (already specified for the SPIR-V Backend) to
  preserve any function attributes

### R5.4 — Verification

The SPIR-V output must pass `spirv-val` for both modes. For AMD vendor mode, validation
must accept the AMD vendor extensions; for Khronos mode, output must validate against
the standard environment.

---

## R6. SPIR-V Translator (SPIR-V → LLVM IR)

### R6.1 — Reverse lowering

The SPIR-V → LLVM IR direction (consumer side) must reconstruct the original
`llvm.convert.*.arbitrary.fp` intrinsic call with the original metadata format string
and rounding mode. This ensures bit-exact round-trip:

```
LLVM IR (intrinsic) → SPIR-V (external call) → LLVM IR (intrinsic, identical)
```

### R6.2 — Recovery from preserved auxdata

When the SPIR-V module has `--spirv-preserve-auxdata` annotations, the translator must
read them and re-create the intrinsic call with original semantics. When auxdata is
absent (e.g., consumer received SPIR-V from a non-LLVM producer), the translator must
fall back to recognizing the call by mangled name and reconstructing operands from the
call signature.

### R6.3 — Pass ordering

The reverse translation must run *before* AMDGPU backend code generation, so that
AMDGPU custom lowering (R2) sees the intrinsic and can lower it to HW instructions, not
the opaque external call.

---

## R7. Validation Requirements

### R7.1 — Round-trip equivalence

For every `(format, src_type, dst_type, rounding, saturate, scale)` tuple:

1. `f32 → format → f32` must match the reference (Triton's existing SW path) within ULP
   bounds appropriate to the format.
2. `IR → SPIR-V → IR → AMDGCN binary` must produce bit-identical machine code to
   `IR → AMDGCN binary` (the SPIR-V hop is semantically transparent).
3. End-to-end: a Triton kernel doing FP8 quantization must produce identical numerical
   output via:
   - Direct AMDGCN compilation
   - SPIR-V (AMD vendor) → consumer → AMDGCN
   - SPIR-V (Khronos) → consumer → AMDGCN (where consumer is responsible for the
     reverse translation)

### R7.2 — Performance baselines

For each architecture, a microbenchmark suite must verify the new intrinsic-based path
matches or beats the current Triton hand-rolled path:

- gfx906: SW expansion only — must be ≤ 1.2× the current SW path runtime (some
  overhead acceptable for code dedup).
- gfx942: must use HW for FNUZ, generic for OCP — must match current Triton path
  runtime within 5%.
- gfx950: must use scaled HW for OCP — must match within 5%.
- gfx1250: must use direct f16 paths and pk8/pk16 wide instructions — must match
  within 5%.

### R7.3 — Coverage matrix

A test matrix must exist with at minimum:

- 5 architectures × 11 format pairs × 5 rounding modes × 2 saturate × 2 scale paths =
  1100 cells
- For each cell: known-good output for representative inputs (zero, denormal, smallest
  normal, largest normal, Inf, NaN, random)
- CI must run the matrix on at least gfx942, gfx950, gfx1250 hardware

### R7.4 — Negative tests

The verifier must reject:

- Unknown format strings (with a descriptive error)
- Unsupported rounding modes for the chosen format
- Saturation flag set on a format with no Inf representation when `saturate=false`
  would also be valid (warning, not error)
- Vector width mismatches between source and result

---

## R8. Migration Plan (non-normative, but recommended)

To make this tractable, deliver in phases:

1. **Phase 1 (current diff is 70% there):** OCP formats unscaled, FP8 HW lowering for
   gfx942/950/1250, generic expansion fallback. Triton continues using its own path.
2. **Phase 2:** Scaled intrinsics + RFC adoption + fused scale DAG combine + FNUZ format
   support. Triton starts emitting scaled intrinsic for MX block paths.
3. **Phase 3:** Stochastic rounding + E8M0FNU + vector-native lowering. Triton's
   `Fp16_to_Fp8E5M2_RTZ`-style paths migrate.
4. **Phase 4:** Delete `ElementwiseOpToLLVM.cpp` SW/HW dispatch table. Triton emits
   intrinsics universally.
5. **Phase 5:** SPIR-V backend Khronos mode + KHR extension proposal.

Phases 1-4 are AMD-only and don't require any SPIR-V work. Phase 5 unblocks the
cross-vendor SPIR-V flow.

---

## Appendix A: References

- `scaled-arbitrary-fp-rfc.md` — RFC for scaled variants (companion document)
- OCP Microscaling Formats MX v1.0 specification:
  https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
- OCP 8-bit Floating Point Specification (OFP8):
  https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-12-01-pdf-1
- Triton AMD `ElementwiseOpToLLVM.cpp` dispatch table — the code being replaced
- LLVM `LegalizeDAG.cpp:3785-4268` — current `CONVERT_TO_ARBITRARY_FP` expansion
- LLVM `SIISelLowering.cpp` `LowerCONVERT_*_ARBITRARY_FP` — current AMDGPU custom lowering

## Appendix B: Glossary

- **OCP**: Open Compute Project. Defines the MX microscaling format family.
- **MX**: Microscaling — block-scaled quantization where N narrow FP values share a
  single E8M0FNU exponent.
- **FNUZ**: Finite, Unsigned Zero — non-IEEE FP variant used by AMD CDNA3 (gfx942).
  No Inf, only one (positive) zero, single NaN encoding.
- **FN**: Finite, NaN-only — IEEE-like but no Inf representation. Used for OCP E4M3FN.
- **RTNE**: Round To Nearest, ties to Even.
- **RTZ**: Round Toward Zero.
- **SR**: Stochastic Rounding.
- **PK**: Packed — instructions that operate on multiple values per VGPR.
