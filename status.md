# LLVM `full-triton-intrinsics` Branch Status

**Branch:** `full-triton-intrinsics` at `b11b8de7f4ba` ("wip on coop matrices")
**Base:** `apfloat-full-for-triton` at `1fc4d8bbb417`
**Date:** 2026-04-10

This file is the canonical handoff document for the LLVM-side work. Two tracks live on the branch: the **arbitrary-FP intrinsics** track (production-ready) and the **cooperative matrix intrinsics** track (design-complete, implementation partial with known bugs).

---

## 0. Current test results (after fixes in working tree)

| Suite | Discovered | Passed | Failed |
|---|---|---|---|
| `check-llvm-codegen-amdgpu` | 4705 | 4699 (+6 XFAIL) | **0** |
| `check-llvm-codegen-spirv` | 1804 | 1727 (+77 unsupported) | **0** |
| `check-llvm-assembler` | 512 | 511 (+1 unsupported) | **0** |
| `check-llvm-verifier` | 388 | 385 (+3 unsupported) | **0** |
| `check-llvm-transforms-instcombine` | 1804 | 1727 (+77 unsupported) | **0** |

## 0.1. Fixes in working tree (uncommitted, needs commit)

| File | Change | Reason |
|---|---|---|
| `llvm/test/CodeGen/AMDGPU/llc-pipeline.ll` | +5 `CHECK-NEXT: Lower cooperative matrix operations to AMDGPU intrinsics` lines | New pass landed in legacy PM pipeline, manifest must match |
| `llvm/test/CodeGen/AMDGPU/llc-pipeline-npm.ll` | +3 `CHECK-NEXT: amdgpu-lower-cooperative-matrix` lines | Same for new pass manager |
| `llvm/test/CodeGen/AMDGPU/float-to-arbitrary-fp-fp8-hw.s` | **DELETED** | Stray committed assembly dump with no RUN line, caused Unresolved test |

These three are the minimum changes to get a clean `ninja check-llvm-codegen-amdgpu` on the branch.

---

## 1. Track A: Arbitrary-FP intrinsics — PRODUCTION-READY

### Intrinsic surface (all in `llvm/include/llvm/IR/Intrinsics.td`)

```
llvm.convert.to.arbitrary.fp.<IntTy>.<FloatTy>(
    FloatTy %val, metadata %format, metadata %round, i1 %saturate)

llvm.convert.from.arbitrary.fp.<FloatTy>.<IntTy>(
    IntTy %bits, metadata %format)

llvm.convert.to.arbitrary.fp.sr.<IntTy>.<FloatTy>(
    FloatTy %val, metadata %format, i32 %seed, i1 %saturate)
```

Plus InstCombine constant folding (from `ConstantInt`/`ConstantFP` inputs).

### Format coverage

| Format | Generic expansion | AMDGPU HW | SPIR-V |
|---|---|---|---|
| Float8E5M2 (OCP) | ✅ | ✅ gfx942/950/1250 | ✅ SPV_EXT_float8 |
| Float8E4M3FN (OCP) | ✅ | ✅ gfx942/950/1250 | ✅ SPV_EXT_float8 |
| Float8E4M3FNUZ | ✅ | ✅ gfx942 native | ❌ no SPIR-V encoding |
| Float8E5M2FNUZ | ✅ | ✅ gfx942 native | ❌ no SPIR-V encoding |
| Float8E4M3B11FNUZ | ✅ | — | ❌ |
| Float8E8M0FNU (MX scale) | ✅ | — (scale-only) | ❌ |
| Float6E3M2FN | ✅ | ✅ gfx950/1250 v32 packed | ❌ no SPIR-V encoding |
| Float6E2M3FN | ✅ | ✅ gfx950/1250 v32 packed | ❌ |
| Float4E2M1FN | ✅ | ✅ gfx950/1250 + FP4 packed fusion | ✅ SPV_INTEL_float4 |
| Float8E4M3 (IEEE) | ❌ TODO | — | — |
| Float8E3M4 | ❌ TODO | — | — |

### AMDGPU HW lowering (`llvm/lib/Target/AMDGPU/SIISelLowering.cpp`)

- `LowerCONVERT_FROM_ARBITRARY_FP` / `LowerCONVERT_TO_ARBITRARY_FP` / `LowerCONVERT_TO_ARBITRARY_FP_SR`
- `performConvertToArbitraryFPCombine` — fuses `convert_to(fdiv(val, scale), fp8)` → scaled HW, gated on E8M0 scale or `arcp` FMF
- `matchFMulArbitraryFPScaleFrom` — fuses `fmul(amdgcn_cvt_*_fp8(...), scale)` → scaled HW (covers legacy ROCDL emission path)
- FROM-direction generic-intrinsic combine — fuses `fmul(convert_from_arbitrary_fp(...), scale)` (covers new generic intrinsic path)
- SR-direction combine — fuses `convert_to_sr(fdiv(val, scale), ..., seed)` → `cvt_scalef32_sr_pk_fp8_f32`
- `performOrCombine` — merges two scalar FP8 `pk_fp8_f32` conversions into one packed; also merges `pk8_fp4` for FP4 nibble pairs
- `hasOCPFP8Semantics()` subtarget helper — disambiguates FNUZ (gfx942) vs OCP (gfx950+/gfx11+) for the same `v_cvt_*_fp8` instruction mnemonics
- FP6 v32 vector paths: `v6i32 ↔ v32{f32,f16,bf16}` via `pk32_*` and `2xpk16_*` intrinsics

### SPIR-V lowering (`llvm/lib/Target/SPIRV/`)

- Three new extensions registered: `SPV_EXT_float8`, `SPV_INTEL_float4`, `SPV_INTEL_fp_conversions`
- New capabilities: `Float8EXT` (4212), `Float4E2M1TypeINTEL` (6212), `FloatConversionsINTEL` (6215)
- New FP encodings: `Float8E4M3EXT` (4214), `Float8E5M2EXT` (4215), `Float4E2M1INTEL` (6214)
- New decoration: `SaturatedToLargestFloat8NormalConversionEXT` (4216) — used for saturating FP8 conversions
- New opcodes: `OpStochasticRoundFToFINTEL` (6217), `OpClampStochasticRoundFToFINTEL` (6218) for SR
- `SPIRVEmitIntrinsics::rewriteConvertArbitraryFPCalls` — pre-IRTranslator pass that replaces metadata format string with i32 ImmArg (APFloat::Semantics enum) because IRTranslator rejects MDString operands
- Instruction selector uses **phantom FP type** approach: builds `OpTypeFloat 8 4214` then emits standard `OpBitcast` + `OpFConvert` sequence
- spirv-val validation wired into test RUN lines

### Gaps in Track A (non-blockers for Triton migration)

- Float8E4M3 (IEEE) and Float8E3M4 formats — not yet in `APFloatBase::getArbitraryFPSemantics`. Triton does not emit these today.
- SPIR-V path is RNE-only for non-SR `OpFConvert`. RTZ kernels going through SPIR-V need IR-side expansion.
- FP6/FNUZ/E8M0 have no SPIR-V encoding in any published extension. These formats must go through the generic bit-manipulation expansion in `LegalizeDAG.cpp` before reaching the SPIR-V backend.

---

## 2. Track B: Cooperative matrix intrinsics — EARLY STAGE

### Intrinsic surface (`llvm/include/llvm/IR/Intrinsics.td` lines 2826-3044)

**Set 1 (baseline):** `load`, `store`, `muladd`, `length`, `construct`
**Set 2 (extended):** `unary`, `binary`, `convert`, `reduce`, `extract`, `insert`, `get_coord`, `prefetch`
**Set 3 (perf):** `muladd_ext`, `muladd_sparse`, `muladd_scaled`, `load_checked`, `store_checked`

All operate on `target("spirv.CooperativeMatrixKHR", elem, scope, rows, cols, use)`.

### 2.1 Critical correctness bugs (from completed layout-verification-agent)

Verified against Triton's `lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp`, which cites ROCm's `amd_matrix_instruction_calculator` as the authoritative source.

**BUG-1: MFMA 32×32 accumulator layout WRONG**
File: `llvm/lib/Target/AMDGPU/AMDGPULowerCooperativeMatrix.cpp` lines 289-301
Current formula emits `row = HalfWave*16 + ((T>>1)*8 + (T&1)*4) + E` which produces contiguous rows 0..15 for lane 0.
Correct formula: `row = HalfWave*4 + T*8 + E`, producing {0..3, 8..11, 16..19, 24..27}.
Exact replacement code in layout-verification-agent report; see `AMDGPULowerCooperativeMatrix.cpp` Section 4d below.

**BUG-2: MFMA 16×16 A-operand layout WRONG**
Same file. `getMFMAElementCoords` does not take `Use` as a parameter. The A operand at `Rows=Cols=16` falls into the accumulator branch and gets the accumulator's `(row=(lane/16)*4+elem, col=lane%16)` formula. Correct for A: `row = lane%16, col = (lane/16)*4 + elem`. (B operand coincidentally matches the accumulator formula at 16×16, so B is accidentally correct.)

**BUG-3: MFMA 32×32×8 A/B operands use wrong generic fallback**
Lines 302-311. Current code treats the per-lane `VecLen` values as flat row-major chunks. This is never the HW layout. Correct:
- A: `row = lane%32, col = (lane/32)*4 + elem`
- B: `row = (lane/32)*4 + elem, col = lane%32`

**FIX direction for BUGS 1-3:** extend `getMFMAElementCoords` signature to accept `unsigned Use`, branch on it, and implement per-Use formulas. Both call sites (`lowerLoad` line 362 and `lowerStore` line 415) already compute `Use`; just pass it through. Exact fix code in the layout-verification-agent report; reproduced in section 2.8 below.

**BUG-4: Non-coopmatrix use replacement in `processFunction` erase loop** (lines ~735-745)
Values that flow into `ret` / non-coopmatrix `store` / function return get `RAUW`'d to undef with TargetExtType because the match-types branch in the erase loop falls through. Happens to work for load→muladd→store chains but breaks any kernel that returns a coop matrix.

**BUG-5: AMDGPU pass silently drops Set 2/3 intrinsics**
`processFunction` collects only `load`/`store`/`muladd`/`length`/`construct` (lines 620-632). Other coopmatrix intrinsics pass through unchanged, their TargetExtType reaches SelectionDAG, and codegen crashes with "Cannot select target extension type". Should either lower them (Set 2/3 implementation) or call `report_fatal_error` / emit a `DiagnosticInfo` like SPIR-V does.

**BUG-6: gfx908 bf16 mis-gate**
The muladd table entries for `bf16×bf16→f32` (16×16×16 and 32×32×8) use `amdgcn_mfma_f32_*bf16_1k` intrinsics that only exist on gfx90a+. Gate is currently `hasMAIInsts() && isWave64()` which includes gfx908. Needs explicit `hasMAIInsts() && getGeneration() >= CDNA2` or equivalent.

**BUG-7: `coopmatrix_length` returns wrong value for FP8 and any future small-element configs**
Lines 442-481 hardcode length based on rows/cols/use only, ignoring element type. Explicitly acknowledged in the code comment. Does not bite today because the muladd table has no FP8 entries; will bite the moment FP8 is added.

**BUG-8: `coopmatrix_muladd_scaled` format enum diverges from `APFloatBase::Semantics`**
The scaled muladd intrinsic defines its own format enum (0=FP8_E5M2, 1=FP8_E4M3, ...) that does not match the APFloat enum used by arbitrary-fp convert. Either align or translate at the lowering site.

### 2.2 Muladd lookup table — only 8 entries

Current `lookupMulAdd` covers (all in `AMDGPULowerCooperativeMatrix.cpp`):

| Config | Gate |
|---|---|
| f16×f16→f32 16×16×16 | WMMA gfx12+/MFMA gfx908+ |
| f16×f16→f32 32×32×8 | MFMA only |
| bf16×bf16→f32 16×16×16 | MFMA (mis-gated, see BUG-6) |
| bf16×bf16→f32 32×32×8 | MFMA (mis-gated) |
| i8×i8→i32 16×16×16 | WMMA/MFMA |
| i8×i8→i32 32×32×8 | MFMA only |

Plus two f16/i8 WMMA v2 entries for gfx12 wave32.

### 2.3 Muladd configs MISSING (required for Triton)

| Missing | ISA |
|---|---|
| fp8×fp8, bf8×bf8, fp8×bf8, bf8×fp8 → f32 at 16×16×32 and 32×32×16 | gfx942, gfx950 |
| fp8 WMMA | gfx1200+, gfx1250 |
| Scaled MX (muladd_scaled) for FP4/FP6/FP8 with E8M0 scale | gfx950, gfx1250 |
| f16/bf16/i8 K=32 variants (WMMA v3) | gfx1250 |
| WMMA v1 (entire family) | gfx1100-1153 (RDNA3) |
| 32×32×16 accumulator shapes | gfx942+ |
| f64×f64→f64 MFMA | gfx90a+ |

### 2.4 Load/store scalarization ⇒ ~5-10× perf regression

Current `lowerLoad`/`lowerStore` emit `VecLen` individual scalar loads per lane with per-element address computation. For a 16×16×16 f16 MFMA that's 64 lanes × 4 elements × 3 operands = 768 scalar loads per muladd, vs Triton's current hand-rolled path which uses ~16 vector loads via `ds_read_b128` / `global_load_dwordx4`.

Fix direction: emit one wide load per lane using the contiguous-in-memory property (elements a lane owns ARE contiguous in the memory layout MFMA/WMMA expects), with fallback to per-element only for layouts where this fails. On gfx12+, prefer `amdgcn_ds_read_tr*` intrinsics that are designed for cooperative matrix loads.

### 2.5 Set 2/3 AMDGPU support ⇒ zero

Set 2 operations (unary/binary/convert/reduce/extract/insert/get_coord/prefetch) and Set 3 (muladd_ext/sparse/scaled/load_checked/store_checked) are defined in Intrinsics.td but the AMDGPU pass doesn't handle them. They crash in SelectionDAG.

**Strategy directive from user: NO element-wise scalarized lowering unless an op genuinely has no HW / cross-lane equivalent.**

Set 2 AMDGPU lowering plan (NOT yet implemented):
- `unary`, `binary`, `convert` on per-lane vectors → plain LLVM IR vector ops (`fneg <4 x f32>`, `fadd`, `fpext`, etc.) — no cross-lane coordination needed because each lane's elements live contiguously in the concrete vector
- `reduce` row/col/2x2 → `amdgcn_ds_swizzle` / `amdgcn_permlane*` / `amdgcn_update_dpp` cross-lane intrinsics, then lane-local reduction
- `extract`/`insert` constant index → `extractelement`/`insertelement` on the concrete vector
- `extract`/`insert` dynamic index → `amdgcn_readlane`/`amdgcn_writelane`
- `get_coord` → arithmetic on lane ID + element index using the Use-aware MFMA/WMMA layout formulas
- `prefetch` → `amdgcn_global_prefetch` or no-op by cache level
- `muladd_ext` → pre/post `fneg` / `fabs` on the concrete vector + regular muladd
- `muladd_scaled` → `amdgcn_cvt_scalef32_*` followed by muladd, or direct scaled MFMA where available (gfx950+)
- `load_checked`/`store_checked` → masked vector loads/stores
- `muladd_sparse` → not yet, requires SWMMAC intrinsics

### 2.6 SPIR-V Set 2/3 support ⇒ only Set 1

Currently `SPIRVInstructionSelector::selectIntrinsic` handles Set 1 and dispatches everything else to `diagnoseUnsupported("cooperative matrix intrinsic not yet supported in SPIR-V backend")` (line ~5001).

**Dispatch plan from completed spirv-research-agent** (all verified against `SPIRVInstrInfo.td` and `SPIRVSymbolicOperands.td` except NV2 opcode numbers which need spec verification):

| Intrinsic | Primary lowering | Extension |
|---|---|---|
| `unary` (Negate/Not) | `OpFNegate`/`OpSNegate`/`OpNot` directly on matrix | `SPV_KHR_cooperative_matrix` (already enabled) |
| `unary` (Abs/Ceil/Floor/Round/Trunc) | `OpCooperativeMatrixPerElementOpNV` + synthesized helper `OpFunction` | `SPV_NV_cooperative_matrix2` (must add) |
| `binary` (Add/Sub/Mul/Div/FRem/bitwise/shifts) | Direct `OpFAdd`/`OpIAdd`/... on matrix | KHR |
| `binary` (Min/Max) | `OpCooperativeMatrixPerElementOpNV` | NV2 |
| `convert` (same Use) | `OpFConvert`/`OpSConvert`/... on matrix | KHR |
| `convert` (Use change) | `OpCooperativeMatrixConvertNV` | NV2 |
| `reduce` (Row/Col/2x2) | `OpCooperativeMatrixReduceNV` + synthesized combiner `OpFunction` | NV2 |
| `reduce` (All) | UNSUPPORTED — no single op; reject or decompose to Row+Col | — |
| `extract` (const idx) | `OpCompositeExtract` | KHR |
| `extract` (dynamic idx) | UNSUPPORTED — no SPIR-V op exists | — |
| `insert` (const idx) | `OpCompositeInsert` | KHR |
| `insert` (dynamic idx) | UNSUPPORTED | — |
| `get_coord` | `OpCooperativeMatrixGetElementCoordINTEL` (6440) — **already in InstrInfo.td** | `SPV_INTEL_joint_matrix` (already registered) |
| `prefetch` | `OpCooperativeMatrixPrefetchINTEL` (6449) — **already in InstrInfo.td** | `SPV_INTEL_joint_matrix` |
| `load_checked` | `OpCooperativeMatrixLoadCheckedINTEL` (6193) — **already in InstrInfo.td** | `SPV_INTEL_joint_matrix` |
| `store_checked` | `OpCooperativeMatrixStoreCheckedINTEL` (6194) — **already in InstrInfo.td** | `SPV_INTEL_joint_matrix` |
| `muladd_ext` (neg_A/B/C + operands) | `OpFNegate` pre-ops + `OpCooperativeMatrixMulAddKHR` | KHR |
| `muladd_ext` (abs_C/clamp) | pre/post `OpCooperativeMatrixPerElementOpNV` + MulAddKHR | NV2 |
| `muladd_sparse` | UNSUPPORTED — no SPIR-V op exists | — |
| `muladd_scaled` | UNSUPPORTED — no SPIR-V op for block-scaled MX MMA | — |

**Critical finding**: four INTEL opcodes for `get_coord`, `prefetch`, `load_checked`, `store_checked` are **already in `SPIRVInstrInfo.td`** but not wired to any LLVM intrinsic. The corresponding selectors just need to be written; no new TableGen work.

**NV2 extension work needed**: add `SPV_NV_cooperative_matrix2` to `SPIRVCommandLine.cpp`, `SPIRVSymbolicOperands.td` (extension + capabilities `CooperativeMatrixReductionsNV`, `CooperativeMatrixConversionsNV`, `CooperativeMatrixPerElementOperationsNV`); add opcodes `OpCooperativeMatrixPerElementOpNV`, `OpCooperativeMatrixReduceNV`, `OpCooperativeMatrixConvertNV` to `SPIRVInstrInfo.td`. All opcode numbers **must be verified against SPIRV-Headers / SPIRV-Registry** before wiring.

### 2.7 Expected Triton pass rate for coop matrix migration TODAY

| Kernel class | Pass rate | Blocker |
|---|---|---|
| Matmul, CDNA, f16×f16→f32 at 16×16×16 | ~0% until BUG-2 fix; ~95% after | layout |
| Matmul, CDNA, f16/bf16 at 32×32×8 | 0% until BUG-1 + BUG-3 + BUG-6 fixes | layout, gate |
| Matmul, CDNA, FP8 | 0% | not in muladd table |
| Matmul, CDNA4, FP4/FP6/MX scaled | 0% | muladd_scaled unimplemented |
| Matmul, RDNA3 (gfx11) | 0% | WMMA v1 entirely absent |
| Matmul, RDNA4 (gfx12), K=16 | ~95% correctness, ~40% perf | scalarized load/store |
| Matmul, RDNA4 (gfx12), K=32 / FP8 | 0% | not in muladd table |
| Attention kernels (GEMM + softmax + GEMM) | 0% | softmax reduction hits Set 2 crash |
| Elementwise on matrix tiles | 0% | Set 2 crash |
| Quantized GEMM via muladd_scaled | 0% | unimplemented |

**Overall coop matrix migration today: ~15-25% of currently-passing Triton tests, ~20-50% of current perf on the subset that works.**

### 2.8 Layout fix — exact code replacement (from layout-verification-agent)

Replace `getMFMAElementCoords` signature and body:

```cpp
static void getMFMAElementCoords(IRBuilder<> &Builder, Value *LaneId,
                                 unsigned Rows, unsigned Cols, unsigned VecLen,
                                 unsigned Use,
                                 SmallVectorImpl<std::pair<Value *, Value *>> &Coords) {
  Value *LaneId64 = Builder.CreateZExt(LaneId, Builder.getInt64Ty());

  if (Use == MatrixA) {
    // MFMA A: row = lane % M, col = (lane / M) * kWidth + elem
    unsigned M = Rows;
    unsigned kWidth = 64 / M;  // 4 for M=16, 2 for M=32
    Value *Row = Builder.CreateURem(LaneId64, Builder.getInt64(M));
    Value *LaneHi = Builder.CreateUDiv(LaneId64, Builder.getInt64(M));
    Value *BaseCol = Builder.CreateMul(LaneHi, Builder.getInt64(kWidth));
    for (unsigned E = 0; E < VecLen; ++E) {
      Value *Col = Builder.CreateAdd(BaseCol, Builder.getInt64(E));
      Coords.push_back({Row, Col});
    }
    return;
  }
  if (Use == MatrixB) {
    // MFMA B: col = lane % N, row = (lane / N) * kWidth + elem
    unsigned N = Cols;
    unsigned kWidth = 64 / N;
    Value *Col = Builder.CreateURem(LaneId64, Builder.getInt64(N));
    Value *LaneHi = Builder.CreateUDiv(LaneId64, Builder.getInt64(N));
    Value *BaseRow = Builder.CreateMul(LaneHi, Builder.getInt64(kWidth));
    for (unsigned E = 0; E < VecLen; ++E) {
      Value *Row = Builder.CreateAdd(BaseRow, Builder.getInt64(E));
      Coords.push_back({Row, Col});
    }
    return;
  }
  // Accumulator (Use == Accumulator)
  if (Rows == 16 && Cols == 16) {
    // row = (lane/16)*4 + elem, col = lane%16  (already correct)
    Value *BaseRow = Builder.CreateMul(
        Builder.CreateUDiv(LaneId64, Builder.getInt64(16)),
        Builder.getInt64(4));
    Value *Col = Builder.CreateURem(LaneId64, Builder.getInt64(16));
    for (unsigned E = 0; E < VecLen; ++E) {
      Value *Row = Builder.CreateAdd(BaseRow, Builder.getInt64(E));
      Coords.push_back({Row, Col});
    }
  } else if (Rows == 32 && Cols == 32) {
    // FIXED: HalfWave stride 4 (not 16); tile stride T*8 (not (T>>1)*8+(T&1)*4)
    Value *HalfWave = Builder.CreateLShr(LaneId64, Builder.getInt64(5));
    Value *HalfWaveOff = Builder.CreateMul(HalfWave, Builder.getInt64(4));
    Value *Col = Builder.CreateURem(LaneId64, Builder.getInt64(32));
    for (unsigned T = 0; T < 4; ++T) {
      Value *TileBase = Builder.CreateAdd(HalfWaveOff,
                                          Builder.getInt64(T * 8));
      for (unsigned E = 0; E < 4; ++E) {
        Value *Row = Builder.CreateAdd(TileBase, Builder.getInt64(E));
        Coords.push_back({Row, Col});
      }
    }
  } else {
    report_fatal_error("unsupported MFMA accumulator shape");
  }
}
```

Both call sites (`lowerLoad`, `lowerStore`) already compute `Use` from `getCoopMatParams`; just forward it.

WMMA v2 accumulator formula is correct. A/B operand support for WMMA v2 is a follow-up; formulas are in the layout-verification-agent report. gfx11 WMMA v1 has a completely different layout (stride-2 in M) and needs a separate `getWMMAv1ElementCoords`.

---

## 3. Task list

Tasks #5 and #14 are completed. Open tasks:

| # | Subject | Blocked by |
|---|---|---|
| 6 | Fix critical correctness bugs in AMDGPULowerCooperativeMatrix pass (BUG-1 through BUG-8) | — |
| 7 | Verify MFMA/WMMA layout formulas | ✅ done (layout-verification-agent report) |
| 8 | Replace scalarized load/store with real vector memory instructions | 6, 7 |
| 9 | Expand muladd lookup table (FP8, FP4/FP6, K=32, scaled, f64) | 6 |
| 10 | Add WMMA v1 (gfx1100-1153) support | 6 |
| 11 | Implement Set 2 on AMDGPU via real HW (no scalarization) | 6 |
| 12 | Implement Set 2/3 SPIR-V lowering via KHR/INTEL/NV extensions | — (research done) |
| 13 | Final check-llvm validation | all above |

---

## 4. Recommended sequencing

1. **Commit the working-tree fixes** (task 14 artifacts) so the branch is clean.
2. **Start Triton migration of elementwise FP conversion paths** (Track A — ready). Replace `ElementwiseOpToLLVM.cpp`'s `Fp8E4M3FN_to_Fp16` lookup-table family with `llvm.convert.*.arbitrary.fp` calls. Validate bit-exact parity with the current `_SW` path on all supported ISAs. Strict perf win on gfx942 (FNUZ HW not currently used by Triton). This unblocks R4-R5 of the requirements doc.
3. **Do NOT migrate matmul paths yet.** Keep Triton on its direct ROCDL MFMA/WMMA emission in `DotOpToLLVM/MFMA.cpp` and `DotOpToLLVM/WMMA.cpp` until the coop matrix track addresses tasks 6-11.
4. **In parallel with the Triton elementwise migration, fix the coop matrix bugs.** Tasks 6 (correctness) and 12 (SPIR-V) are independent and can run in parallel. Once 6 lands, tasks 8-11 unblock.
5. **Revisit matmul migration** after all coop matrix tasks are complete and numerical-correctness tests pass end-to-end.

---

## 5. Reference documents

- `/home/mrsidims/LLVM/llvm-project/arbitrary-fp-intrinsics-requirements.md` — full R1-R8 requirements for the arbitrary-fp track
- `/home/mrsidims/LLVM/llvm-project/scaled-arbitrary-fp-rfc.md` — scaled intrinsic family RFC
- Layout-verification-agent report (ephemeral, summarized in section 2.1 and 2.8 above)
- SPIR-V research-agent report (ephemeral, summarized in section 2.6 above)
- Triton reference: `/home/mrsidims/Triton/triton/lib/Dialect/TritonGPU/IR/LinearLayoutConversions.cpp` — authoritative MFMA/WMMA layouts, cites `amd_matrix_instruction_calculator`
- Triton reference: `/home/mrsidims/Triton/triton/third_party/amd/lib/TritonAMDGPUToLLVM/ElementwiseOpToLLVM.cpp` — current hand-rolled FP8 conversion paths to be replaced in R4
- Triton reference: `/home/mrsidims/Triton/triton/third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/MFMA.cpp` — current hand-rolled matmul path (keep using until coop matrix track is ready)

---

## 6. Implementation plan for remaining work

### Phase 1 — Commit the green baseline

Commit the three working-tree changes (llc-pipeline.ll, llc-pipeline-npm.ll, deletion of float-to-arbitrary-fp-fp8-hw.s). The branch must be at zero AMDGPU/SPIR-V/Verifier/Assembler/InstCombine failures before any agent starts modifying code.

### Phase 2 — Correctness fixes (task 6, unblocks everything else)

One agent owning `AMDGPULowerCooperativeMatrix.cpp`:
- Apply layout fix from section 2.8 (BUGS 1-3): extend `getMFMAElementCoords` signature with `Use`, add Use-aware branches, thread `Use` through call sites.
- Fix BUG-4: in the erase loop, replace non-match-type RAUW(undef) with RAUW to the concrete vector for all uses, then erase.
- Fix BUG-5: in `processFunction`, detect Set 2/3 intrinsics and either lower them (if task 11 lands in same pass) or emit a clear `DiagnosticInfo` error matching SPIR-V's behavior.
- Fix BUG-6: add `Subtarget->hasBF16MFMAv1Insts()` helper or explicit gfx90a+ gate for bf16 entries.
- Fix BUG-7: either add element-type operand to `coopmatrix_length` or have `lowerLength` defer to the muladd lookup to derive the true length. Note: the former requires an intrinsic signature change and is a user-visible API change; the latter is a workaround.
- Fix BUG-8: align `coopmatrix_muladd_scaled` format enum with `APFloatBase::Semantics` (translate at lowering site is acceptable).
- Add numerical-correctness test that runs a small kernel end-to-end and compares against a reference computation (the existing tests only check IR shapes).

### Phase 3 — Parallel tracks (after Phase 2)

**3a. Muladd table expansion (task 9)**
Agent owning `lookupMulAdd` in `AMDGPULowerCooperativeMatrix.cpp`. Add entries for:
- FP8/BF8 MFMA on gfx942 (FNUZ semantics), gfx950 (OCP via `hasOCPFP8Semantics()`)
- f16/bf16/i8 K=32 MFMA on gfx942+
- f64×f64 MFMA on gfx90a+
- 32×32×16 accumulator shapes
- Scaled MX (`coopmatrix_muladd_scaled`) for FP4/FP6/FP8 on gfx950/1250 mapping to `v_mfma_scale_*` / scaled WMMA
- Reference Triton's `third_party/amd/lib/TritonAMDGPUTransforms/MfmaGroup.cpp` and `WmmaGroup.cpp` for the full HW matrix Triton currently supports.

**3b. WMMA v1 support for gfx11 (task 10)**
Agent adds a separate code path in `getConcreteVectorType`, `lookupMulAdd`, and a new `getWMMAv1ElementCoords` function. WMMA v1 has `kRegister` M stride 2 (not 1) and `kLane` M bit at position 1 (not 8). Reference: Triton `AMDWmmaEncodingAttr::getTileLayout` lines 715-720.

**3c. Vectorize load/store (task 8)**
Agent rewrites `lowerLoad`/`lowerStore` to emit one wide load per lane using the contiguous-in-memory property of MFMA/WMMA lane layouts. On gfx12+ prefer `amdgcn_ds_read_tr*` for shared-memory loads. Fall back to scalar only for layouts where lane elements are genuinely non-contiguous. Verify lane-contiguity per shape×use combination against the ISA.

**3d. Set 2 AMDGPU implementation (task 11)**
Agent adds lowering for `unary`/`binary`/`convert` (vector ops), `reduce` (cross-lane via `amdgcn_ds_swizzle`/`permlane`/DPP), `extract`/`insert` (element access with constant/dynamic index paths via `amdgcn_readlane`/`writelane`), `get_coord`, `prefetch`, `muladd_ext` (pre/post fneg/fabs + muladd), `load_checked`/`store_checked` (masked loads). NO element-wise scalarization per user directive.

**3e. SPIR-V Set 2/3 implementation (task 12)**
Agent implements the dispatch table from section 2.6. Work splits into:
- Wire existing INTEL opcodes (`get_coord`, `prefetch`, `load_checked`, `store_checked`) — no TableGen changes needed, just new selector functions
- KHR element-wise path for `unary`/`binary`/`convert` (native SPIR-V ops on coop matrix operands where the KHR spec allows)
- NV2 extension registration (extension, capabilities, opcodes) for `OpCooperativeMatrixReduceNV`, `OpCooperativeMatrixConvertNV`, `OpCooperativeMatrixPerElementOpNV` — **must verify opcode numbers against SPIRV-Headers**
- Helper-function synthesis for per-element and reduce ops (`OpCooperativeMatrixPerElementOpNV` takes an `OpFunction` id)
- `diagnoseUnsupported` for genuinely unsupported ops (muladd_sparse, muladd_scaled, dynamic-index extract/insert, reduce-all)

### Phase 4 — Final validation (task 13)

One agent runs the full LLVM check suite, confirms zero regressions vs baseline, and files any follow-up tasks for failures that require design discussion.

---

## 7. For the implementing agents

Each agent working on a Phase 3 task should:
1. Read `/home/mrsidims/LLVM/llvm-project/status.md` (this file) for the authoritative state summary
2. Read `/home/mrsidims/LLVM/llvm-project/arbitrary-fp-intrinsics-requirements.md` for the broader requirements context
3. Read `/home/mrsidims/LLVM/llvm-project/llvm/lib/Target/AMDGPU/AMDGPULowerCooperativeMatrix.cpp` and its tests before making changes
4. NEVER use element-wise scalarized fallback without an explicit comment explaining why no HW equivalent exists
5. Write tests that verify real HW instructions appear in the output (grep for `v_mfma_*`, `v_cvt_pk_*`, `ds_read_b*_tr`, `OpCooperativeMatrix*`, etc.) — not just IR shape matches
6. Add at least one numerical-correctness test that runs end-to-end and compares against a reference
7. Run `ninja check-llvm-codegen-amdgpu` and `ninja check-llvm-codegen-spirv` before marking their task complete
8. Update their task status via TaskUpdate and leave a summary of what changed in the task description
