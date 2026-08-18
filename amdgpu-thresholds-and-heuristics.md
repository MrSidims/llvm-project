# AMDGPU Backend — Thresholds, Heuristics and Tuning Knobs

An inventory of every numeric threshold, cost constant and heuristic in
`llvm/lib/Target/AMDGPU`, classified by *where the number comes from*:
hardware, subtarget features, an exposed knob, or a frozen literal in the
source. The goal is to answer: **how much of the AMDGPU cost/heuristic model is
hardcoded, and which of those constants are worth promoting to per-kernel
tunables?**

Tree state: `main` @ `68e94c904067`. All `file:line` anchors are against that
revision.

## Coverage — what is exhaustive and what is not

Read this before treating any absence in this document as evidence.

| Axis | Status | Method |
|---|---|---|
| `cl::opt` flags | **Exhaustive (159)** | machine-extracted, whole-directory, verified count |
| Per-function attributes | **Exhaustive for tunable ones (~20)** | grep over `getFnAttribute*`, `hasFnAttribute`, `getInteger*Attribute` |
| `AMDGPUTargetTransformInfo.{h,cpp}` | **Exhaustive** | both files read end to end |
| `SISchedule.td` | **Exhaustive** — and it is the *only* sched model in the backend | verified: no other `.td` defines `SchedMachineModel`/latencies |
| Named `const`/`constexpr` numeric constants | **Exhaustive within the filter (108 after excluding FP-expansion coefficients, bitfield masks, and the hazard/waitcnt files)** | full uncapped sweep |
| Scheduler, RA, promote-alloca, IGroupLP, perf-hint, clause/cluster | **Thorough** | read directly |
| **Bare inline numeric literals** | **NOT exhaustive** | ~2000 comparison literals and ~600 non-`const` local inits exist directory-wide; the large majority are bit widths, operand indices and encoding fields, but the residue contains real policy (this is how `SIFixSGPRCopies`'s `Score < 3` was initially missed) |
| **R600 path** | **Covered** | `R600InstrInfo`, `R600ISelLowering`, `R600MachineScheduler`, `R600MachineCFGStructurizer` read; legacy target, so recorded but not ranked for promotion |
| **Instruction-selection / legalizer / libcall thresholds** | **Covered** | `AMDGPUInstructionSelector` (7693 lines), `AMDGPULegalizerInfo` (8727), `AMDGPUInstCombineIntrinsic`, `AMDGPULibCalls`, `AMDGPUISelDAGToDAG`, `AMDGPULowerBufferFatPointers`, `AMDGPURegBankLegalizeHelper`, `AMDGPUAsmPrinter`, `GCNRegPressure` audited — overwhelmingly type-legality, but three real policy sites found (`selectBITOP3`, src-modifier `MaxDepth`, pow cutoff) |
| **MC layer, AsmParser, Disassembler** | **Excluded by design** | encoding constants, T0 by construction |

Practical consequence: the **knob inventory (Part V) is complete** — you can
trust "if it is not in that table, there is no flag for it". The
**frozen-constant catalogue (Parts I–IV) is high-coverage but still not
*provably* complete**; it captures every named constant, every subsystem that
matters for performance, and now every file over ~1500 lines in the backend, but
a bare inline literal in a small unread pass could still escape it. The residue
after the read pass is much smaller than the raw ~2000-literal figure suggests —
the files audited last accounted for ~40k of the directory's lines and yielded
only three new policy sites, which is the expected saturation curve.

Companion document: [`amdgpu-backend-autotuning.md`](amdgpu-backend-autotuning.md)
covers a narrower, deeper dive into two specific subsystems (the CoExec
scheduler and the MFMA VGPR/AGPR rewrite stage) with a phased tuning plan. This
document is the broad inventory; it deliberately does not repeat that analysis.

---

## 0. TL;DR — headline findings

1. **159 `cl::opt` flags** exist in the AMDGPU backend. Of those, only **~46 are
   numeric or enum tuning knobs**; the rest are boolean pass on/off switches,
   debug dumps, or test hooks. **116 of the 159 are `cl::Hidden`.**
2. **Only 9 knobs are reachable per-function via an IR attribute.** Everything
   else is `-mllvm`, i.e. whole-module. This is the single biggest structural
   obstacle to per-kernel autotuning — you cannot tune two kernels in one TU
   differently without recompiling.
3. **The TTI cost model is almost entirely frozen literals.** Target dependence
   enters only through a small number of subtarget predicates
   (`hasFullRate64Ops`, `hasFastFMAF32`, `hasTrigReducedRange`,
   `hasPackedFP32Ops`, `has16BitInsts`, …) that *select between* hardcoded
   integers. There is not a single `cl::opt` in the arithmetic or intrinsic cost
   paths. The instruction-count estimates (e.g. "f32 exp = 13 or 17 full-rate
   ops") are transcriptions of the lowering sequences, not measurements.
4. **The user's premise is confirmed and is worse than "context dependent"**:
   `GCNTTIImpl::getArithmeticInstrCost` returns `TCC_Free` for an `FMUL` whose
   single user is an `FADD`/`FSUB` with contract. The cost of an instruction
   therefore depends on (a) the surrounding IR, (b) the *denormal mode* of the
   enclosing function, (c) per-instruction fast-math flags, and (d) a global
   `TargetOptions.AllowFPOpFusion`. Same opcode, same subtarget, four different
   answers. See §3.4. The selector shows the same pattern independently:
   `selectBITOP3` fuses at ≥2 matched ops when the value is **divergent** but
   demands ≥4 when it is **uniform**, the difference being a hand-estimated
   charge for SGPR↔VGPR moves — a *divergence-context*-dependent profitability
   threshold written as two bare literals. Its comment also records that
   TableGen's `AddedComplexity` could not express the decision, which is a
   recurring reason policy ends up frozen in C++.
5. **The machine model (`SISchedule.td`) is explicitly labelled as guessed.**
   `WriteBarrier = 500` carries the comment `XXX: Guessed ???`; the file header
   says the latencies "may not be accurate"; `MispredictPenalty = 20` is a
   deliberate hack against early-ifcvt.
6. **Highest-value promotion candidates** (ranked in §9): the GCN scheduler's
   `ErrorMargin`/`HighRP*Bias`/`MaxVGPRPressureInc`, the IGroupLP exact-shape
   gate, `getInliningThresholdMultiplier() == 11`, the TTI spill thresholds
   26/32, `getMaxInterleaveFactor() == 8`, and the loop-alignment cutoffs
   64/128/192.
7. **Do not tune**: `GCNHazardRecognizer` wait-state tables, `SIInsertWaitcnts`
   counter limits, register allocation granules. These are correctness
   constraints that happen to be written as integers. §5.3.

---

## 1. Taxonomy — four tiers of "constant"

Every number in this document falls into exactly one tier. The tier determines
whether tuning it is *safe*, *meaningful*, or *possible*.

| Tier | Name | Origin | Tunable? | Example |
|---|---|---|---|---|
| **T0** | Hardware invariant | ISA encoding / hardware timing | **No — correctness** | `GCNHazardRecognizer` wait states, `getVGPRAllocGranule()` |
| **T1** | Subtarget-derived | `GCNSubtarget` feature bits / TableGen | No (implied by `-mcpu`) | `getWavefrontSize()`, `hasFullRate64Ops()` |
| **T2** | Exposed knob | `cl::opt` and/or IR attribute | **Yes** | `amdgpu-unroll-threshold-private=2700` |
| **T3** | Frozen literal | Hardcoded in `.cpp`/`.h`/`.td` | Only by recompiling | `getInliningThresholdMultiplier() → 11` |

T3 is the interesting tier: these are policy decisions with no escape hatch. A
large fraction of them were introduced with a single benchmark's number and have
never been revisited.

A fifth, hybrid category appears in a handful of well-designed passes: a
constant that is **T3 by default but T2 by opt-in**, using the
attribute-then-flag precedence pattern documented in §10. Only five knobs in the
whole backend use it.

---

# PART I — TargetTransformInfo

`AMDGPUTargetTransformInfo.{h,cpp}` — 1912 lines, two implementations
(`AMDGPUTTIImpl` shared/R600-era, `GCNTTIImpl` for GCN).

## 2. The rate abstraction

All arithmetic costs are expressed as multiples of four helper functions.
`AMDGPUTargetTransformInfo.h:83-110`:

| Helper | Throughput value | CodeSize value | Tier |
|---|---|---|---|
| `getFullRateInstrCost()` | `TCC_Basic` = **1** | 1 | T3 |
| `getHalfRateInstrCost()` | **2** | 2 | T3 |
| `getQuarterRateInstrCost()` | **4** | 2 | T3 |
| `getTransInstrCost()` | = quarter rate (**4**) | 2 | T3 |
| `get64BitInstrCost()` | `hasFullRate64Ops() ? full : hasHalfRate64Ops() ? half : quarter` | same | **T1** |

`get64BitInstrCost` is the *only* rate helper with any target dependence.

The header carries an unresolved TODO on the quarter-rate value:

> `// TODO: The size is usually 8 bytes, but takes 4x as many cycles. Maybe`
> `// should be 2 or 4.`

That is a live admission that the throughput/code-size conflation is unresolved.
For a code-size-driven autotuning objective this constant matters directly.

**Note that these numbers are *relative rates*, not cycles.** They encode
"a quarter-rate op costs 4x a full-rate op", which is true across all GCN
generations — but they cannot express, for example, that transcendental *latency*
differs from transcendental *throughput*, or that gfx950 VALU is uniformly
1 cycle. This is a structural limitation, not a bad constant.

## 3. Cost tables

### 3.1 `getArithmeticInstrCost` — `AMDGPUTargetTransformInfo.cpp:527-690`

| Case | Cost formula | Tier | Notes |
|---|---|---|---|
| `SHL`/`SRA`/`SRL`, i64 | `get64BitInstrCost` | T1 | |
| `AND`/`OR`/`XOR`, i64 | `2 * full` | T3 | "split into 2x 32-bit" |
| `MUL` i64 | `4*quarter + 4*full` (= 20) | T3 | expansion estimate |
| `MUL` i16/i32 | `quarter` unless `has16BitInsts` | T1/T3 | |
| `FDIV` f64 | `7*get64Bit + quarter + 3*half` (+`3*full` if `!hasUsableDivScaleConditionOutput`) | T1+T3 | |
| `FDIV` f16 (with 16-bit insts) | `4*full + 2*trans` | T3 | |
| `FDIV` f32 **afn** | `trans + full` | T3 | **context**: keyed on `CxtI->hasApproxFunc()` |
| `FDIV` f32/f16 generic | `(f16?14:10)*full + 1*trans`, `+2*full` if `!HasFP32Denormals` | T3 + **mode** | |
| `FMUL` fused | `TCC_Free` | T3 | **context** — see §3.4 |
| `FADD`/`FSUB`/`FMUL` | `NElts` halved if `hasPackedFP32Ops`/`hasBF16PackedInsts` | T1 | |
| `FNEG` | `TLI->isFNegFree(SLT) ? 0 : NElts` | T1 | |

The literals `14`, `10`, `7`, `4`, `3`, `2` are transcriptions of the number of
machine instructions each expansion emits. They are *accurate as counts* and
*unvalidated as costs* — no weighting for the fact that `V_RCP_F32` has ~4x the
latency of `V_ADD_F32` beyond the coarse rate bucket.

### 3.2 `getIntrinsicInstrCost` — `AMDGPUTargetTransformInfo.cpp:735-900`

| Intrinsic | f32 cost | Tier | Feature gate |
|---|---|---|---|
| `exp`/`exp2`/`exp10` f64 | `NumOps = 20` (+1 exp10, +3 exp) | T3 | |
| `exp`/`exp2` f32, `!afn` | `NumFullRateOps = hasFastFMAF32() ? 13 : 17` | **T1** | |
| `exp10` f32 `!afn` | same, +1 | T1 | |
| any f32 exp, denormals on | `+5` | **mode** | `HasFP32Denormals` |
| `log`/`log10` f32, `!afn` | `hasFastFMAF32() ? 8 : 11` | **T1** | |
| `log2` f32 | 0 full-rate + 1 trans | T3 | pure `v_log_f32` |
| any f32 log, denormals on | `+5` | **mode** | |
| `sin`/`cos` f32 | `hasTrigReducedRange() ? 2 : 1` full-rate + trans | **T1** | |
| `sqrt` f32 `!afn` | `HasFP32Denormals ? 17 : 16` | **mode** | afn → 0 full-rate |
| `minimumnum`/`maximumnum` | `NumOps = 3`, or `1` if IEEE mode off | **mode** | reads `amdgpu-ieee` attribute at `:1799` |
| `abs` | `2 * full` | T3 | |

`hasFastFMAF32` toggling 13↔17 and 8↔11 is genuine target dependence. The `+5`
denormal penalty is a single constant applied uniformly to exp *and* log, which
is suspicious — the mode-switch sequences differ.

### 3.3 Control flow, vector, shuffle

**`getCFInstrCost`** — `:1350-1380`. All T3:

```
const int CBrCost = SCost ? 5 : 7;      // conditional branch
Br         → CBrCost
UncondBr   → SCost ? 1 : 4              // "about 4 slots on gfx900"
Switch     → (NumCases + 1) * (CBrCost + 1)
Ret        → SCost ? 1 : 10
```

The "gfx900" comment is a tell: measured once, on one chip, applied to every
target from gfx6 to gfx12.

**`getVectorInstrCost`** — `:1420-1470`. All T3: dynamic index → `2`; `i16`
index 0 → `0`; `i8` extract with ≥4 power-of-two elements → `0`; ≥32-bit
element → `0` (free, sub-register access).

**`getShuffleCost`** — `EltsPerReg = 32 / ScalarSize`; broadcast → `1`;
generic case modelled on `v_perm_b32`. T3, but the `32` is the VGPR width, which
is T1-by-nature and invariant across all AMDGPU targets.

### 3.4 Context dependence — the FMUL/FMA case

`AMDGPUTargetTransformInfo.cpp:584-604`:

```cpp
case ISD::FMUL:
  // Check possible fuse {fadd|fsub}(a,fmul(b,c)) and return zero cost for
  // fmul(b,c) supposing the fadd|fsub will get estimated cost for the whole
  // fused operation.
  if (CxtI && CxtI->hasOneUse())
    if (const auto *FAdd = dyn_cast<BinaryOperator>(*CxtI->user_begin())) {
      const int OPC = TLI->InstructionOpcodeToISD(FAdd->getOpcode());
      if (OPC == ISD::FADD || OPC == ISD::FSUB) {
        if (ST->hasMadMacF32Insts() && SLT == MVT::f32 && !HasFP32Denormals)
          return TargetTransformInfo::TCC_Free;
        if (ST->has16BitInsts() && SLT == MVT::f16 && !HasFP64FP16Denormals)
          return TargetTransformInfo::TCC_Free;

        const TargetOptions &Options = TLI->getTargetMachine().Options;
        if (Options.AllowFPOpFusion == FPOpFusion::Fast ||
            (FAdd->hasAllowContract() && CxtI->hasAllowContract()))
          return TargetTransformInfo::TCC_Free;
      }
    }
```

`getArithmeticInstrCost(FMUL, f32)` can return **1** or **0** depending on:

1. whether `CxtI` is supplied at all (many callers pass `nullptr`);
2. whether the fmul has exactly one use;
3. what that use is;
4. the fast-math flags on *both* instructions;
5. the function's denormal mode;
6. a whole-module `TargetOptions` setting.

Consequences that matter for tuning:

- **The value is not a function of (opcode, type, subtarget).** Any autotuner
  that assumes cost is a pure function of those three will mis-model.
- **It is caller-dependent.** A pass that queries per-type (SLP's
  `getArithmeticInstrCost(Instruction::FMul, VecTy)` with no `CxtI`) gets 1; a
  pass that queries per-instruction gets 0. Two passes disagree about the cost
  of the same IR.
- **It double-counts against itself.** The discount assumes the FADD will be
  charged for the whole FMA, but `case ISD::FADD` falls through to the generic
  path and is charged as a plain add. The pair therefore costs `0 + 1 = 1` where
  an actual `v_fma_f32` costs 1 — correct in aggregate, but only if both halves
  are queried with context.

This is a known live problem; see [`tti-fma-fusion-cost-status.md`](tti-fma-fusion-cost-status.md)
and the two candidate branches recorded in the project memory.

Other context/mode-dependent costs found:

| Site | Depends on | Kind |
|---|---|---|
| `:657` FDIV f32 | `CxtI->hasApproxFunc()` | per-instruction fast-math flag |
| `:603` FMUL | `Options.AllowFPOpFusion` | whole-module codegen option |
| `HasFP32Denormals`, `HasFP64FP16Denormals` | function's `denormal-fp-math` attribute | **per-function** |
| `:1799` `getArithmeticInstrCost` min/maxnum | `amdgpu-ieee` attribute | **per-function** |
| `ICA.getFlags().approxFunc()` (exp/log/sqrt) | call-site fast-math | per-instruction |

The per-function ones are the good news: denormal mode and IEEE mode already
give TTI a legitimate per-kernel input channel. They are the existence proof
that per-kernel cost modelling is architecturally possible here.

### 3.5 Register/vector shape

| Query | Value | Tier |
|---|---|---|
| `getNumberOfRegisters(RCID)` | **`return 4;`** | T3 |
| `getRegisterBitWidth` scalar | 32 | T1-natural |
| `getRegisterBitWidth` vector | `128` if `hasPackedFP64Ops()‖hasPackedU64Ops()`, `64` if `hasPackedFP32Ops()`, else `32` | T1 |
| `getMinVectorRegisterBitWidth()` | 32 | T3 |
| `getMaximumVF(ElemWidth, Opcode)` | load/store: `32*4/ElemWidth`; else `4`/`2`/`2`/`2`/`1` by width | T3 |
| `getLoadVectorFactor`/`getStoreVectorFactor` | hard **128-bit** cap | T3 |
| `getLoadStoreVecRegBitWidth` | `512` global/constant/buffer, `8*getMaxPrivateElementSize()` private, else `128` | T3/T1 |
| `getMaxInterleaveFactor` | `1` if scalar else **`8`** | T3 |
| `getNumberOfParts` | `divideCeil(EltCount - 1, 4)` | T3 |
| `getTypeLegalizationCost` | `if (Size <= 256) return Cost; Cost.first += (Size + 255)/256;` | T3 |

`getNumberOfRegisters` deserves a callout — the implementation is literally:

```cpp
unsigned GCNTTIImpl::getNumberOfRegisters(unsigned RCID) const {
  // NB: RCID is not an RCID. In fact it is 0 or 1 for scalar or vector
  // registers. See getRegisterClassForType for the implementation.
  // In this case vector registers are not vector in terms of
  // VGPRs, but those which can hold multiple values.

  // This is really the number of registers to fill when vectorizing /
  // interleaving loops, so we lie to avoid trying to use all registers.
  return 4;
}
```

Four. For a chip with 256 addressable VGPRs. The reasoning is register-pressure
conservatism — the comment says outright "we lie to avoid trying to use all
registers" — but it means every unroller and vectorizer downstream is reasoning
about a fictitious 4-register machine. **This is the single most consequential
frozen constant in AMDGPU TTI**, and also the riskiest to change — see §9.

`getMaxInterleaveFactor() == 8` is the second: it drives loop interleaving for
*every* GCN target regardless of the actual occupancy the kernel will achieve.

### 3.6 Unrolling — `getUnrollingPreferences`, `:110-330`

The best-instrumented area of AMDGPU TTI.

| Knob | Default | Exposure |
|---|---|---|
| `UP.Threshold` | **300** | **`amdgpu-unroll-threshold` attribute** (T2, per-function!) |
| `amdgpu-unroll-threshold-private` | 2700 | `cl::opt` hidden |
| `amdgpu-unroll-threshold-local` | 1000 | `cl::opt` hidden |
| `amdgpu-unroll-threshold-if` | 200 | `cl::opt` hidden |
| `amdgpu-unroll-runtime-local` | true | `cl::opt` hidden |
| `amdgpu-unroll-max-block-to-analyze` | 32 | `cl::opt` hidden |
| `UP.BEInsns += 3` | 3 | T3 |
| `MaxAlloca = (256 - 16) * 4` | 960 | T3 |
| `dependsOnLocalPhi` recursion depth | `Depth < 10` | T3 |
| LDS unroll inhibit | `LocalGEPsSeen > 1 ‖ L->getLoopDepth() > 2` | T3 |
| `UP.MaxIterationsCountToAnalyze` | 32 | T3 |

The `amdgpu-unroll-threshold` attribute is the **model to copy**: a per-function
integer attribute read directly by TTI with a hardcoded fallback. It is
documented in `AMDGPUUsage.rst` and is exactly the mechanism a per-kernel
autotuner needs. It is one of only two TTI-level per-function knobs.

### 3.7 Inlining

| Site | Value | Tier |
|---|---|---|
| `getInliningThresholdMultiplier()` | **`return 11;`** (`.h:262`) | T3 |
| `InlinerVectorBonusPercent` | `0` (`.h:75`) | T3 |
| `amdgpu-inline-arg-alloca-cost` | 4000 | T2 hidden |
| `amdgpu-inline-arg-alloca-cutoff` | 256 | T2 hidden |
| `amdgpu-inline-max-bb` | 1100 | T2 hidden |
| `NrOfSGPRUntilSpill` | **26** (`:1593`) | T3 |
| `NrOfVGPRUntilSpill` | **32** (`:1594`) | T3 |
| `ArgStackCost` | 1 | T3 |
| `getCallerAllocaCost` single-BB bonus | `Threshold += Threshold / 2` | T3 |

`adjustInliningThresholdUsingCallee` uses 26/32 as the point at which it starts
charging for register pressure. Those are the *only* occupancy-awareness in the
inliner, they are per-wave-count-agnostic, and they are frozen. A kernel
targeting 1 wave/EU and a kernel targeting 8 get the same numbers.

`11` as the inlining threshold multiplier is a pure magic number. It is a *large*
multiplier (X86 uses 1) and it is the reason AMDGPU inlines so aggressively.
Making it tunable is cheap and the search space is small.

### 3.8 Memory, LSR, prefetch

| Query | Value | Tier |
|---|---|---|
| `getMaxMemIntrinsicInlineSizeThreshold()` | **1024** (both impls) | T3 |
| `amdgpu-memcpy-loop-unroll` | 16 — *"based on microbenchmarks on gfx1030"* | T2 hidden |
| `getMemcpyLoopLoweringType` | `I32EltsInVector = 4`, `<MemcpyLoopUnroll*4 x i32>`, constant length only | T3 |
| `getCacheLineSize()` | `ST->getDataCacheLineSize()` **only if the target has prefetch insts**, else 0 | T1 |
| `getPrefetchDistance()` | `ST->hasPrefetch() ? 128 : 0` | T3 gated by T1 |
| `getScalingFactorCost` | gfx1250 `scale_offset` fold ≤16 bytes → 0, else 1 | T1+T3 |
| `isLSRCostLess` | custom ordering, `Insns + ScaleCost` compared first | T3 policy |
| `isNumRegsMajorCostOfLSR()` | `false` | T3 policy |
| `shouldDropLSRSolutionIfLessProfitable()` | `true` | T3 policy |
| `preferSLPInstCountCheck()` | `!hasGFX940Insts() && !hasGFX950Insts()` | **T1** |

`amdgpu-memcpy-loop-unroll = 16` is the most honest constant in the backend —
its comment names the chip it was measured on. It is also, for that exact reason,
a prime autotuning target on every *other* chip.

`preferSLPInstCountCheck()` is an interesting example of target dependence
expressed as a *policy* switch rather than a number: SLP's instruction-count
heuristic is disabled on gfx940/gfx950 because MFMA-heavy code wants
vectorization the count check would reject.

## 4. TTI classification summary

| Category | Count (approx) | Autotuning relevance |
|---|---|---|
| Frozen literals (T3) | **~75** | High — the bulk of the model |
| Subtarget-gated selections (T1) | ~20 | None directly; implied by `-mcpu` |
| `cl::opt` knobs (T2, module-wide) | **9** | Medium — cannot vary per kernel |
| Per-function attribute knobs | **2** (`amdgpu-unroll-threshold`, `amdgpu-ieee`) | **High — the right mechanism** |
| Context/flag-dependent | ~6 sites | Confounds any static cost table |

**Roughly 85% of AMDGPU's cost model is unreachable without recompiling LLVM.**

---

# PART II — Scheduling and register pressure

## 5. GCN machine scheduler

### 5.1 `GCNSchedStrategy.{h,cpp}` frozen constants

| Constant | Value | Location | Meaning |
|---|---|---|---|
| `ErrorMargin` | **3** | `GCNSchedStrategy.h` | RP slack subtracted from SGPR/VGPR limits to absorb tracker imprecision |
| `HighRPSGPRBias` | **7** | `.h:130` | extra SGPR pressure penalty in the high-RP stage |
| `HighRPVGPRBias` | **7** | `.h:133` | ditto for VGPRs |
| `ScheduleMetrics::ScaleFactor` | **100** | `.cpp:133` | `getMetric() = BubbleCycles*100/ScheduleLength` |
| `ScoredRemat::FreqInfo::ScaleFactor` | **1024** | `.h:590` | block-frequency quantisation for remat scoring |
| `MaxVGPRPressureInc` | **16** | `.cpp:390` | how much VGPR pressure a candidate may add — **carries two FIXMEs** |
| long-latency tie-break | `TryCand.SU->Latency > 10 * Cand.SU->Latency` | `.cpp:964-966` | 10x latency ratio to prefer memory ops |

`MaxVGPRPressureInc = 16` is worth quoting because the code itself flags it:

```cpp
  // FIXME: This is very inaccurate.
  const unsigned MaxVGPRPressureInc = 16;
```

`ErrorMargin = 3` is a global fudge factor that directly costs registers on
every region in every kernel. On a kernel at an occupancy cliff, 3 registers is
the difference between 8 and 7 waves.

### 5.2 Scheduler knobs (T2)

| Flag | Default | Effect |
|---|---|---|
| `amdgpu-schedule-metric-bias` | **10** | weight of occupancy vs latency; 100 = occupancy only |
| `amdgpu-scheduler-pending-queue-limit` | 256 | max `Available+Pending` before pending queue is inspected |
| `amdgpu-vgpr-threshold-percent` | 0 | scales *both* Critical and Excess RP limits back by a percentage |
| `amdgpu-schedule-relaxed-occupancy` | false | lets memory-bound/wave-limited kernels drop the occupancy target |
| `amdgpu-use-amdgpu-trackers` | false | AMDGPU-specific RP trackers |
| `amdgpu-disable-unclustered-high-rp-reschedule` | false | disable a scheduling stage |
| `amdgpu-disable-clustered-low-occupancy-reschedule` | false | disable a scheduling stage |
| `amdgpu-disable-rewrite-mfma-form-sched-stage` | **true** | the MFMA VGPR/AGPR rewrite stage is **off by default** |
| `amdgpu-sched-strategy` | `""` | select a custom strategy; also readable as the `amdgpu-sched-strategy` **attribute** |
| `amdgpu-post-sched-strategy` | — | attribute-only, post-RA |

`amdgpu-schedule-metric-bias` is the closest thing the backend has to a single
"tuning dial" and its default of 10 out of 100 is unexplained. The profit
formula at `.cpp:2179-2185` uses it directly.

`amdgpu-sched-strategy` and `amdgpu-post-sched-strategy` are notable as the only
*scheduler* knobs already readable per function.

### 5.3 Occupancy and RP limit derivation — `initialize()`

```
TargetOccupancy = RelaxedOcc ? MFI.getMinAllowedOccupancy() : MFI.getOccupancy()
VGPRBudget      = alignDown(Addressable / TargetOccupancy, Granule)      // known-excess-RP path
VGPRExcessLimit = (VGPRThresholdPercentOpt * VGPRExcessLimit + 99) / 100  // if flag set
else            = limit - VGPRLimitBias - ErrorMargin
```

The `Granule` and `Addressable` terms come from `AMDGPU::IsaInfo` (T0/T1). The
**occupancy target is a step function** — VGPR budget jumps discontinuously as
`TargetOccupancy` changes — which is why naive continuous tuning of RP knobs
produces cliff behaviour. This is analysed at length in
[`amdgpu-backend-autotuning.md`](amdgpu-backend-autotuning.md).

### 5.4 `SIMachineScheduler.cpp` — the legacy scheduler

Entirely T3, and self-described as arbitrary:

| Constant | Value | Meaning |
|---|---|---|
| `Cand.SGPRUsage > 60` | 60 | comment says *"arbitrary limit of 60"* |
| `VregCurrentUsage > 120` | 120 | VGPR pressure switch point |
| group size | `NumHighLatencies <= 6 → 2`, `<= 12 → 3`, else `4` | high-latency clustering |
| `SubGraph.size() > 5` | 5 | subgraph size bail-out |

This scheduler is not the default, which is why nobody has revisited these. It
is reachable via `-misched=si` (`SISchedRegistry` at `AMDGPUTargetMachine.cpp:827`)
or the `si-scheduler` subtarget feature (`ST.enableSIScheduler()`, `:1411`) —
note *not* via `amdgpu-sched-strategy`, which selects among the GCN strategies.

### 5.5 `AMDGPUIGroupLP.cpp` — SchedGroup pipelining

The most aggressively hardcoded heuristic in the backend. `MFMAExpInterleaveOpt`
gates its entire strategy on an **exact shape match**:

```cpp
bool IsSmallKernelType =
    MFMAEnablement == 2 && ExpRequirement == 4 && TransPipeCount == 32;
bool IsLargeKernelType =
    MFMAEnablement == 4 && ExpRequirement == 4 && TransPipeCount == 64;
if (!(IsSmallKernelType || IsLargeKernelType))
  return false;
```

Two shapes. Everything else silently falls off the optimisation. The pipeline
offset formulas are equally fixed:

```
(1 + UsesVALU) * MFMARatio * (I + 1)
(2 + UsesFMA) * (ExpRequirement - 1) + 1
I / 4 + 1
MissPenalty = (ProblemSize / 2) + 1
```

`MFMASmallGemmOpt` hardcodes the DS:MFMA interleave ratio as `{DS,2} + {MFMA,1}`
repeated `MFMACount * 3` times.

Exposed knobs are only about the *solver*, not the *policy*:
`amdgpu-igrouplp-exact-solver` (false), `-cutoff` (0), `-max-branches` (0),
`-cost-heur` (true, with a comment saying results are mixed and it "should be
set on a case-by-case basis" — i.e. an explicit invitation to autotune).

### 5.6 `SISchedule.td` — the machine model

Header comment: *"The latency numbers are taken from AMD Accelerated Parallel
Processing guide. They may not be accurate."*

| Resource | Latency | Comment in source |
|---|---|---|
| `MicroOpBufferSize` | 1 | in-order |
| `IssueWidth` | 1 | |
| `MispredictPenalty` | 20 | *"FIXME: Approximate 2 * branch cost. Try to hack around bad early-ifcvt heuristics"* |
| `WriteBranch` | 8 | |
| `WriteExport` | 4 | |
| `WriteLDS` | 5 | *"Can be between 2 and 64"* |
| `WriteSALU` | 1 | |
| `WriteSMEM` | 5 | |
| `WriteVMEM` | 80 | |
| `WriteBarrier` | 500 | **`XXX: Guessed ???`** |
| DGEMM/MAI passes | 2 / 4 / 8 / 16 | per-pass MFMA classes |
| `ReadAdvance<MIVGPRRead>` | −2 | |
| `ReadAdvance<MIMFMARead>` | −4 | |

Everything a list scheduler does rests on this table. `WriteVMEM = 80` is a
single number for L1 hit, L2 hit, and HBM miss. `WriteBarrier = 500` is admitted
to be a guess. These are T3-in-TableGen, meaning they are not even reachable by
`-mllvm`.

### 5.7 Other scheduling-adjacent latencies

| Constant | Value | Location |
|---|---|---|
| `FenceLatency` | **2000** | `AMDGPUBarrierLatency.cpp:95` |
| `amdgpu-barrier-signal-wait-latency` | 16 | `AMDGPUBarrierLatency.cpp:32` (T2 hidden) |
| `MaskLatencyBoost` | 3 | `AMDGPUHazardLatency.cpp:43` |
| `Dep.setLatency(1)` for tensorcnt/asynccnt | 1 | `GCNSubtarget::adjustSchedDependency` |

A synthetic `FenceLatency` of 2000 exists purely to make the scheduler hoist work
across fences. It is a policy lever masquerading as a latency, and it is frozen.

---

# PART III — Register allocation, spilling, clauses, hazards

## 6. Register allocation and spilling

| Constant / knob | Value | Location | Tier |
|---|---|---|---|
| `amdgpu-num-vgprs-for-wwm-alloc` | 10 | `SILowerSGPRSpills.cpp:51` | T2 |
| `amdgpu-spill-sgpr-to-vgpr` | true | `SIRegisterInfo.cpp:32` | T2 |
| `amdgpu-spill-vgpr-to-agpr` | true | `SIFrameLowering.cpp:28` | T2 |
| `amdgpu-stress-{vgpr,sgpr,agpr}` | 0 | `SIRegisterInfo.cpp:43-51` | T2 (test) |
| `sgpr-regalloc-npm` / `vgpr-regalloc-npm` / `wwm-regalloc-npm` | Default | `AMDGPUTargetMachine.cpp:259-267` | T2 |
| `amdgpu-num-sgpr` / `amdgpu-num-vgpr` | subtarget max | `GCNSubtarget.cpp:544,624` | **attribute** (deprecated) |
| `amdgpu-agpr-alloc` | `{~0u,~0u}` | `GCNSubtarget.cpp:674` | **attribute** |
| gfx90a default AGPR split | `MinNumAGPRs = MaxNumAGPRs = MaxVectorRegs / 2` | `GCNSubtarget.cpp:679` | **T3** |
| AGPR alignment | `alignTo(MinNumAGPRs, 4)` | `GCNSubtarget.cpp:683` | T0 (accum_offset granularity) |
| `amdgpu-mfma-vgpr-form` | true | `SIMachineFunctionInfo.cpp:36` | T2 + **attribute** |
| `amdgpu-dynamic-vgpr-block-size` | 0 | `Utils/AMDGPUBaseInfo.cpp:2472` | **attribute** |

The gfx90a **"default to splitting half the registers if AGPRs are required"**
is a pure T3 policy with a TODO next to it:

```cpp
// TODO: it shall be possible to estimate maximum AGPR/VGPR pressure and split
//       register file accordingly.
```

For an MFMA kernel this 50/50 split is often exactly wrong, and the only escape
is to hand-write `amdgpu-agpr-alloc` — which does exist as a per-function
attribute, making this one of the better-served areas.

`AMDGPUNextUseAnalysis` deserves special mention: it ships **two entire
heuristic presets** (`"graphics"` and `"compute"`) selected by
`amdgpu-next-use-analysis-config`, with four individual override flags
(`-count-phis`, `-forward-only`, `-precise-use-modeling`,
`-use-preheader-model`) each following the `getNumOccurrences()` precedence
pattern. The source comment is candid:

> `// FIXME: Hopefully we will soon converge on a single way of calculating`
> `// next-use distance and remove these presets.`

This is the *only* place in the backend where a full alternative heuristic
configuration is switchable at runtime. It is a good template for
"profile-selected heuristic families" if autotuning goes that direction.

## 7. Memory clauses and clustering

| Constant / knob | Value | Location | Exposure |
|---|---|---|---|
| `amdgpu-max-memory-clause` | **15** | `SIFormMemoryClauses.cpp:30` | flag **+ attribute** (`:275`) |
| `amdgpu-hard-clause-length-limit` | attribute default **255** | `SIInsertHardClauses.cpp:46,199` | flag **+ attribute** |
| `DefaultMemoryClusterDWordsLimit` | **8** | `SIInstrInfo.h:42` | **+ attribute** `amdgpu-max-memory-cluster-dwords` |
| `MaxAddressRegs` | `12 + 1 + 1` | `SILoadStoreOptimizer.cpp:110` | T3 (encoding-derived) |
| `ScanLimit` | 12 | `SIInstrInfo.cpp:11267` | T3 (compile time) |
| `MaxNonSmrdLoadSize` | 128 | `AMDGPURegisterBankInfo.cpp:1050` | T3 |
| `MaxRegisterSize` | 1024 | `AMDGPULegalizerInfo.cpp:56` | T3 |

`amdgpu-max-memory-clause`'s comment explains its value physically — *"Clauses
longer than 15 instructions would overflow one of the counters and stall"* —
making 15 a **T0/T1 boundary**, not a free parameter. Note it is still lower on
targets with smaller counters, so the constant is an upper bound rather than a
tuned value.

The three clause/cluster knobs are the **best-designed knobs in the backend**:
flag *and* per-function attribute, with the flag taking precedence only when
explicitly set. Copy this.

## 8. Hazards — DO NOT TUNE

`GCNHazardRecognizer.cpp` contains ~60 named wait-state constants
(`SmrdSgprWaitStates = 4`, `MFMA32x32WritesAGPRAccVgprReadWaitStates = 18`,
`WMMAWaitStates[] = {5,9,3,5,9,17,2}`, …). `SIInsertWaitcnts.cpp` derives its
counter limits from `AMDGPU::HardwareLimits(IV)`.

**These are T0.** They encode hardware pipeline timing that must be respected or
the program computes wrong results. They are correctly hardcoded and should
never appear in an autotuning search space.

Two exceptions in that file *are* tuning knobs:

- `amdgpu-mfma-padding-ratio` (0) — "fill a percentage of the latency between
  neighbouring MFMA with `s_nop`s". A genuine performance dial.
- `amdgpu-snop-padding` (0) — insert `s_nop x` before every instruction. Debug.
- `amdgpu-wmma-vnop-hoisting` (true) — hoist WMMA hazard `V_NOP`s out of loops.

`AMDGPUWaitSGPRHazards` similarly exposes `amdgpu-sgpr-hazard-boundary-cull`,
`-mem-wait-cull`, and `-mem-wait-cull-threshold` (8), each also readable as a
per-function attribute. These trade compile-time-visible hazard conservatism for
wait-state count — legitimately tunable.

---

# PART IV — IR-level and codegen passes

## 9. Per-pass constant catalogue

### `AMDGPUPromoteAlloca.cpp` — the reference implementation

```cpp
void AMDGPUPromoteAllocaImpl::setFunctionLimits(const Function &F) {
  const int R600MaxVectorRegs = 16;
  MaxVectorRegs = F.getFnAttributeAsParsedInteger(
      "amdgpu-promote-alloca-to-vector-max-regs",
      IsAMDGCN ? PromoteAllocaToVectorMaxRegs : R600MaxVectorRegs);
  if (PromoteAllocaToVectorMaxRegs.getNumOccurrences())
    MaxVectorRegs = PromoteAllocaToVectorMaxRegs;
  VGPRBudgetRatio = F.getFnAttributeAsParsedInteger(
      "amdgpu-promote-alloca-to-vector-vgpr-ratio",
      PromoteAllocaToVectorVGPRRatio);
  if (PromoteAllocaToVectorVGPRRatio.getNumOccurrences())
    VGPRBudgetRatio = PromoteAllocaToVectorVGPRRatio;
}
```

| Constant | Value | Exposure |
|---|---|---|
| `amdgpu-promote-alloca-to-vector-max-regs` | 32 | flag + attribute |
| `amdgpu-promote-alloca-to-vector-vgpr-ratio` | 4 | flag + attribute |
| `amdgpu-promote-alloca-to-vector-limit` | 0 (bytes) | flag |
| `promote-alloca-vector-loop-user-weight` | 4 | flag |
| `R600MaxVectorRegs` | 16 | T3 |
| non-entry function VGPR cap | `32u` unless alwaysinline | T3 |
| `UserScore` | `1 + LoopUserWeight * LI.getLoopDepth(...)` | T3 formula |
| `MaxElements` | `(MaxVectorRegs * 32) / eltBits`, reject `< 2` | derived |
| `AllocaCost` | `Size * 8`, drawn from `VectorizationBudget` | T3 |

This pass is the gold standard: **two of its four numeric policies are
per-function attributes with flag override, and both are documented in
`AMDGPUUsage.rst`.** Every promotion candidate in §11 should be implemented this
way.

### `AMDGPUPerfHintAnalysis.cpp` — heuristic that *sets* attributes

| Knob | Default | Meaning |
|---|---|---|
| `amdgpu-membound-threshold` | 50 | % of instructions that are memory ops → set `amdgpu-memory-bound` |
| `amdgpu-limit-wave-threshold` | 50 | % → set `amdgpu-wave-limiter` |
| `amdgpu-indirect-access-weight` | 1000 | weight of an indirect access |
| `amdgpu-large-stride-weight` | 1000 | weight of a large-stride access |
| `amdgpu-large-stride-threshold` | 64 | what counts as "large stride" |
| `GlobalMemAccPercentage > 50` | 50 | T3, `:271` — hardcoded twin of the flag above |

This pass is architecturally the most interesting for autotuning: it is a
*heuristic that emits per-function attributes* (`amdgpu-memory-bound`,
`amdgpu-wave-limiter`) which downstream passes consume. It is already the bridge
between "analysis" and "per-kernel policy". Note the hardcoded `> 50` at `:271`
that duplicates `MemBoundThresh` — the same policy expressed twice, once tunable
and once not.

### `AMDGPUCodeGenPrepare.cpp`

`amdgpu-codegenprepare-break-large-phis` (true) / `-threshold` (32) /
`-force-break-large-phis`; `-mul24` (true); `-expand-div64` (false);
`-disable-idiv-expansion` / `-disable-fdiv-expansion`;
`-widen-constant-loads` (false). All T2, all module-wide.

### `AMDGPUAtomicOptimizer.cpp`

`ActiveLanesThreshold = 5` (`:664`) — T3. `amdgpu-atomic-optimizer-strategy`
(`ScanOptions::Iterative`) — T2 enum, DPP vs Iterative vs None.

### `AMDGPUSplitModule.cpp`

`amdgpu-module-splitting-max-depth` (8, note `O(2^N)`),
`-large-threshold` (2.0f), `-merge-threshold` (0.7f). Three *float* knobs — the
only floats in the backend's tuning surface besides
`amdgpu-long-branch-factor` (1.0).

### `SIISelLowering::getPrefLoopAlignment`

```
CacheLineAlign = Align(64)          // Align(32) on GFX950 (fetch window)
LoopSize >  192  → bail
LoopSize <=  64  → no alignment
LoopSize <= 128  → cache-line align
else             → S_INST_PREFETCH, 2 behind / 1 ahead
```

Gated on `hasInstPrefetch()` / `hasInstFwdPrefetchBug()` and disabled by
`amdgpu-disable-loop-alignment`. The 64/128/192 cutoffs are T3; the 64→32
GFX950 adjustment is T1.

### `SIFixSGPRCopies.cpp` — VGPR→SGPR copy lowering

An entire cost/benefit heuristic with no exposed knob at all:

```cpp
unsigned Penalty =
    Info->NumSVCopies + Info->SiblingPenalty + Info->NumReadfirstlanes;
unsigned Profit = Info->SChain.size();
Info->Score = Penalty > Profit ? 0 : Profit - Penalty;
Info->NeedToBeConvertedToVALU = Info->Score < 3;
```

| Constant | Value | Location | Kind |
|---|---|---|---|
| VALU-conversion score cutoff | **3** | `:1071` | T3 policy |
| M0 init insertion search window | **50** | `:611` | T3 compile-time |

The `Score < 3` decides whether a whole scalar chain gets moved to the vector
unit. It is an unweighted `Profit - Penalty` with three terms summed at equal
weight and a bare integer cutoff — the most tunable-looking heuristic in the
backend that has never been given a flag.

### `SIInstrInfo` — clustering tiers and scan budgets

`shouldClusterMemOps` derives its behaviour from `MaxMemoryClusterDWords`
(default 8, per-function attribute) via `NumDWords = ((LoadSize+3)/4)*ClusterSize`,
which produces the tier table spelled out in its own comment:

```
(1) 1  <= LoadSize <= 4  : cluster at max 8 mem ops
(2) 5  <= LoadSize <= 8  : cluster at max 4 mem ops
(3) 9  <= LoadSize <= 12 : cluster at max 2 mem ops
(4) 13 <= LoadSize <= 16 : cluster at max 2 mem ops
(5) LoadSize >= 17       : do not cluster
```

The comment says the heuristic came from "observations and performance related
experiments", and the function immediately below carries
`// FIXME: This behaves strangely.` about the interleaving of 32 load+stores.
Since the tiers are *derived* from a tunable attribute, this is one of the few
places where per-kernel tuning already reaches a real policy — worth noting as a
success case.

Scan budgets (T3, compile-time, not performance): `MaxInstScan = 20` (`:10788`,
`:10832`), `MaxUseScan = 10` (`:10815`), `ScanLimit = 12` (`:11267`).

### `SIOptimizeExecMasking.cpp`

`InstLimit = 25` (`:333`, exec-copy search window), `SearchLimit = 5` (`:462`).
Both T3 compile-time budgets.

### Instruction selection and legalization

Both large files were audited. `AMDGPULegalizerInfo.cpp` (8727 lines) is
essentially pure type-legality — its one global constant, `MaxRegisterSize =
1024` (`:56`), is the physical register-file width and is T0/T1, not policy.
`AMDGPUInstructionSelector.cpp` (7693 lines) is mostly type/bank dispatch but
contains three genuine policy sites:

**`selectBITOP3` — `AMDGPUInstructionSelector.cpp:4463-4480`.** The most
interesting cost decision in the selector, because it carries *two different
profitability thresholds keyed on divergence*:

```cpp
if (NumOpcodes < 2 || Src.empty())
  return false;
...
if (NumOpcodes == 2 && IsB32) {
  // Avoid using BITOP3 for OR3, XOR3, AND_OR. This is not faster but makes
  // asm more readable. This cannot be modeled with AddedComplexity because
  // selector does not know how many operations did we match.
  ...
} else if (NumOpcodes < 4) {
  // For a uniform case threshold should be higher to account for moves
  // between VGPRs and SGPRs. ...
  return false;
}
```

Three things are notable. The divergent path fuses at ≥2 matched ops, the
uniform path demands ≥4 — the extra 2 is a hand-estimated charge for the
`v_readfirstlane` plus SGPR↔VGPR moves, and it is a bare literal. The `NumOpcodes
== 2` bail is explicitly **cosmetic** ("not faster but makes asm more readable")
— a readability heuristic living in the instruction selector. And the comment
records that TableGen's `AddedComplexity` mechanism *could not express this*,
which is exactly the kind of pressure that pushes policy into hardcoded C++.

**Source-modifier search depth — `:5267` and `:5284`.** `getSrcStats()` and
`getLastSameOrNeg()` both walk the def chain looking for neg/hi/lo modifiers to
fold into packed-fp16 operands, bounded by a default argument `int MaxDepth = 3`.
A modifier chain longer than 3 is silently not folded. No flag, no attribute.

**`NumGroups >= 4`** at `:3920` is tensor-load group structure, not a threshold.

### IR-level libcall expansion — `AMDGPULibCalls.cpp`

`pow`/`powr`/`pown(x, c)` with integral `c` is expanded to a multiply chain when
`|c| <= 12` (`:985`), gated on `isUnsafeFiniteOnlyMath`. Two observations:

- The constant has **zero target dependence** — the same 12 applies to every
  GPU. It is a frozen constant living in a target-specific pass.
- **The comment is stale and the threshold is mis-sized.** The comment at `:982`
  says `pow(x,c) = (x*x*..x); where ... the number of x == c`, implying a linear
  chain of `c-1` multiplies. The loop below it is binary exponentiation:

  ```cpp
  while (abs_opr1 > 0) {
    valx2 = valx2 ? B.CreateFMul(valx2, valx2, "__powx2") : opr0;
    if (abs_opr1 & 1)
      nval = nval ? B.CreateFMul(nval, valx2, "__powprod") : valx2;
    abs_opr1 >>= 1;
  }
  ```

  So `pow(x,12)` costs **4** fmuls, not 11. The cutoff of 12 was sized against a
  cost model the code no longer implements — the real cost is `~2·log2(c)`, so a
  much larger cutoff would be justified on the stated reasoning. Worth a
  correctness-preserving follow-up independent of any tuning work.

`numArgs > 3` at `:1979` is a signature-shape check, not policy.

### SelectionDAG lowering overrides — `AMDGPUISelLowering.cpp`

AMDGPU overrides two generic `TargetLowering` tuning fields. Both are frozen,
both are invisible in any flag listing because they are plain field assignments:

| Field | AMDGPU | Generic default | Location |
|---|---|---|---|
| `MaxGluedStoresPerMemcpy` | 16 | **0** | `:74` |
| `GatherAllAliasesMaxDepth` | 16 | **18** (`TargetLoweringBase.cpp:1036`) | `:623` |

The first turns on store-ganging in memcpy lowering that is off by default
everywhere else. The second *lowers* the generic alias-gathering budget, and its
comment is an explicit admission of a workaround: "when we can more precisely
specify load legality per address space, we should be able to make
FindBetterChain/MergeConsecutiveStores smarter so that they can figure out what
to do in 2 iterations without all N > 4 stores on the same chain."

### GlobalISel combiner iteration budget

All three AMDGPU combiners set `CInfo.MaxIterations = 1` with the comment
"Disable fixed-point iteration to reduce compile-time"
(`AMDGPUPreLegalizerCombiner.cpp:270`, `AMDGPUPostLegalizerCombiner.cpp:503`,
`AMDGPURegBankCombiner.cpp:688`). `MaxIterations == 0` would mean iterate to
fixed point, so this is a real compile-time/quality tradeoff — second-order
combines are never seen.

**Not an AMDGPU-specific gap, though**: AArch64, RISCV, X86, WebAssembly and
SPIR-V all do the same. This belongs in the inventory for completeness but is
*not* a promotion candidate — changing it is an LLVM-wide GlobalISel policy
question, not a per-kernel tuning knob.

### R600 (legacy VLIW path)

Deprecated pre-GCN target, included here to close the coverage gap rather than
because it is worth tuning. It is nonetheless the most *honestly documented*
heuristic code in the backend.

**`R600MachineScheduler.cpp`** — the ALU/TEX clause-switch heuristic cites its
source:

```cpp
// We use the heuristic provided by AMD Accelerated Parallel Processing
// OpenCL Programming Guide :
// The approx. number of WF that allows TEX inst to hide ALU inst is :
// 500 (cycles for TEX) / (AluFetchRatio * 8 (cycles for ALU))
...
unsigned NeededWF = 62.5f / ALUFetchRationEstimate;
```

`62.5f` is `500 / 8` precomputed — a vendor-documented hardware ratio frozen as
a float literal. Alongside it: `getWFCountLimitedByGPR()` returns `248 /
GPRCount` (`:49`, register-file size), `InstKindLimit[IDOther] = 32` (`:34`),
`OccupiedSlotsMask = 31` (VLIW5), and a GPR estimate `NearRegisterRequirement =
2 * Available[IDFetch].size()` carrying `TODO : use RegisterPressure`.

**`R600MachineCFGStructurizer.cpp:387-388`** — block-cloning cost model, two
frozen thresholds combined multiplicatively:

```cpp
unsigned BlockSizeThreshold = 30;
unsigned CloneInstrThreshold = 100;
...
return ((BlkSize > BlockSizeThreshold) &&
    (BlkSize * (MBB->pred_size() - 1) > CloneInstrThreshold));
```

**`R600InstrInfo.cpp:861-891`** — all three `isProfitableToIfCvt` overloads
`return true` unconditionally and `isProfitableToUnpredicate` returns `false`,
ignoring every parameter passed to them (`NumCycles`, `ExtraPredCycles`,
`BranchProbability`). A degenerate heuristic: always if-convert, never
un-predicate. `R600ISelLowering.cpp`'s `512 + 4096*n` ladder is the constant-bank
address map — ABI, not policy.

### Miscellaneous frozen constants

| Constant | Value | Location |
|---|---|---|
| `Threshold` (S_SET_GPR_IDX scan) | 20 | `SIPreEmitPeephole.cpp:810` |
| `SearchLimit` | 16 | `SIShrinkInstructions.cpp:726` |
| `MaxUses` | 10 | `AMDGPURewriteOutArguments.cpp:118` |
| `HighPriority` | 3 | `AMDGPUSetWavePriority.cpp:120` |
| `VALU_MAX` / `TRANS_MAX` / `SALU_CYCLES_MAX` | 5 / 4 / 4 | `AMDGPUInsertDelayAlu.cpp:91-99` (T0, encoding) |
| `NextUse` loop-depth shift | `7 * Depth` | `AMDGPUNextUseAnalysis.h:169` |
| `MaxDynamicVGPRBlocks` | 8 | `Utils/AMDGPUBaseInfo.h:274` |
| `WAVE32_NOPS` / `WAVE64_NOPS` | 4 / 8 | `AMDGPUWaitSGPRHazards.cpp:155-156` (T0) |
| use-scan `Limit` | 10 | `AMDGPUISelDAGToDAG.cpp:4675,4714` |
| `MaxReorderWindow` | 6 | `GCNILPSched.cpp:167` (depth-spread cutoff, sits next to `bool const DisableSchedCriticalPath = false` — a dead knob) |
| src-modifier fold `MaxDepth` | 3 | `AMDGPUInstructionSelector.cpp:5267,5284` |
| `isCanonicalized`/`denormalsEnabled` walk `MaxDepth` | 5 | `SIISelLowering.h:571,573` |
| noalias-propagation `MaxDepth` | 5 | `AMDGPULowerModuleLDSPass.cpp:1475` |
| `MaxNumLanes` (image-op return width) | 4 | `SIISelLowering.cpp:1459` |
| `MaxAsyncMarks` | 16 | `SIInsertWaitcnts.cpp:806` (T0, counter width) |
| `MaxGluedStoresPerMemcpy` | 16 (generic 0) | `AMDGPUISelLowering.cpp:74` |
| `GatherAllAliasesMaxDepth` | 16 (generic 18) | `AMDGPUISelLowering.cpp:623` |
| combiner `MaxIterations` | 1 (0 = fixed point) | `AMDGPU{Pre,Post}LegalizerCombiner`, `AMDGPURegBankCombiner` — LLVM-wide norm |
| pow multiply-chain cutoff | `\|c\| <= 12` | `AMDGPULibCalls.cpp:985` (no target dependence; comment stale, see above) |
| `MaxRegisterSize` | 1024 | `AMDGPULegalizerInfo.cpp:56` (T0/T1) |
| `computeKnownBits` recursion cutoff | `Depth >= 6` / `> 6` | `SIISelLowering.cpp:14868,14938` |
| `MaxStoresPerMemset/Memcpy/Memmove` | `~0U` | `AMDGPUISelLowering.cpp` |
| `setSchedulingPreference` | `Sched::RegPressure` | `AMDGPUISelLowering.cpp` |
| `setJumpIsExpensive(true)` | — | `AMDGPUISelLowering.cpp` |
| `setMinFunctionAlignment` | `Align(4)` | `SIISelLowering.cpp:200` |
| `setPrefFunctionAlignment` | `Align(getInstCacheLineSize())` | `SIISelLowering.cpp:201` (T1) |
| `amdgpu-assume-external-call-stack-size` | 16384 | `AMDGPUResourceUsageAnalysis.cpp:41` |
| `amdgpu-assume-dynamic-stack-object-size` | 4096 | `AMDGPUResourceUsageAnalysis.cpp:46` |
| `amdgpu-indirect-call-specialization-threshold` | 3 | `AMDGPUAttributor.cpp:27` |
| `amdgpu-set-wave-priority-valu-insts-threshold` | 100 | `AMDGPUSetWavePriority.cpp:28` |
| `amdgpu-max-return-arg-num-regs` | 16 | `AMDGPURewriteOutArguments.cpp:69` |
| `amdgpu-s-branch-bits` | 16 | `SIInstrInfo.cpp:57` |
| `amdgpu-nsa-threshold` | 2 (clamped ≥2) | `GCNSubtarget.cpp:52` + attribute |

---

# PART V — Complete knob inventory

## 10. Numeric and enum `cl::opt` knobs (46)

These are the flags whose value is a *quantity*, i.e. the actual tuning surface.
All are `-mllvm` and module-wide unless the Attr column says otherwise.

| Flag | Default | Hidden | Attr | Subsystem |
|---|---|---|---|---|
| `amdgpu-assume-dynamic-stack-object-size` | 4096 | y | | resource usage |
| `amdgpu-assume-external-call-stack-size` | 16384 | y | | resource usage |
| `amdgpu-atomic-optimizer-strategy` | `Iterative` | n | | atomics |
| `amdgpu-barrier-signal-wait-latency` | 16 | y | | sched latency |
| `amdgpu-codegenprepare-break-large-phis-threshold` | 32 | n | | codegen prep |
| `amdgpu-force-generic-version` | 0 | n | | metadata (test) |
| `amdgpu-hard-clause-length-limit` | *(255 via attr)* | y | **y** | clauses |
| `amdgpu-igrouplp-exact-solver-cutoff` | 0 | y | | IGroupLP |
| `amdgpu-igrouplp-exact-solver-max-branches` | 0 | y | | IGroupLP |
| `amdgpu-indirect-access-weight` | 1000 | y | | perf hint |
| `amdgpu-indirect-call-specialization-threshold` | 3 | n | | attributor |
| `amdgpu-inline-arg-alloca-cost` | 4000 | y | | TTI inline |
| `amdgpu-inline-arg-alloca-cutoff` | 256 | y | | TTI inline |
| `amdgpu-inline-max-bb` | 1100 | y | | TTI inline |
| `amdgpu-kernarg-preload-count` | 0 | n | | kernarg |
| `amdgpu-large-stride-threshold` | 64 | y | | perf hint |
| `amdgpu-large-stride-weight` | 1000 | y | | perf hint |
| `amdgpu-limit-wave-threshold` | 50 | y | | perf hint |
| `amdgpu-long-branch-factor` | 1.0 | y | | long branch reg |
| `amdgpu-lower-module-lds-strategy` | `hybrid` | y | | LDS |
| `amdgpu-max-memory-clause` | 15 | y | **y** | clauses |
| `amdgpu-max-return-arg-num-regs` | 16 | y | | out args |
| `amdgpu-membound-threshold` | 50 | y | | perf hint |
| `amdgpu-memcpy-loop-unroll` | 16 | y | | TTI memcpy |
| `amdgpu-mfma-padding-ratio` | 0 | y | | hazard/perf |
| `amdgpu-module-splitting-large-threshold` | 2.0f | y | | split module |
| `amdgpu-module-splitting-max-depth` | 8 | n | | split module |
| `amdgpu-module-splitting-merge-threshold` | 0.7f | y | | split module |
| `amdgpu-nsa-threshold` | 2 | y | **y** | MIMG NSA |
| `amdgpu-num-vgprs-for-wwm-alloc` | 10 | n | | WWM regalloc |
| `amdgpu-promote-alloca-to-vector-limit` | 0 | n | | promote alloca |
| `amdgpu-promote-alloca-to-vector-max-regs` | 32 | n | **y** | promote alloca |
| `amdgpu-promote-alloca-to-vector-vgpr-ratio` | 4 | n | **y** | promote alloca |
| `amdgpu-s-branch-bits` | 16 | n | | branch range (debug) |
| `amdgpu-sched-strategy` | `""` | y | **y** | scheduler |
| `amdgpu-schedule-metric-bias` | 10 | y | | scheduler |
| `amdgpu-scheduler-pending-queue-limit` | 256 | y | | scheduler |
| `amdgpu-set-wave-priority-valu-insts-threshold` | 100 | y | | wave priority |
| `amdgpu-sgpr-hazard-mem-wait-cull-threshold` | 8 | y | **y** | SGPR hazards |
| `amdgpu-snop-padding` | 0 | y | | hazard (debug) |
| `amdgpu-stress-agpr` | 0 | y | | RA stress |
| `amdgpu-stress-sgpr` | 0 | y | | RA stress |
| `amdgpu-stress-vgpr` | 0 | y | | RA stress |
| `amdgpu-unroll-max-block-to-analyze` | 32 | y | | TTI unroll |
| `amdgpu-unroll-threshold-if` | 200 | y | | TTI unroll |
| `amdgpu-unroll-threshold-local` | 1000 | y | | TTI unroll |
| `amdgpu-unroll-threshold-private` | 2700 | y | | TTI unroll |
| `amdgpu-vgpr-threshold-percent` | 0 | y | | scheduler RP |
| `promote-alloca-vector-loop-user-weight` | 4 | n | | promote alloca |
| `amdhsa-code-object-version` | COV6 | y | | ABI |

## 11. Boolean pass and policy switches (~113)

Grouped by subsystem; all `cl::opt<bool>`. These matter for autotuning as
*binary* dimensions (pass on/off is a legitimate search axis) but carry no
threshold information.

**Pass enable/disable** (`AMDGPUTargetMachine.cpp`, mostly hidden):
`amdgpu-attributor-enable`, `amdgpu-dce-in-ra`, `amdgpu-dpp-combine`,
`amdgpu-early-ifcvt`, `amdgpu-early-inline-all`, `amdgpu-enable-delay-alu`,
`amdgpu-enable-hipstdpar`, `amdgpu-enable-image-intrinsic-optimizer`,
`amdgpu-enable-lower-exec-sync`, `amdgpu-enable-lower-module-lds`,
`amdgpu-enable-object-linking`, `amdgpu-enable-pre-ra-optimizations`,
`amdgpu-enable-promote-kernel-arguments`,
`amdgpu-enable-remove-incompatible-functions`,
`amdgpu-enable-rewrite-partial-reg-uses`, `amdgpu-enable-sw-lower-lds`,
`amdgpu-enable-uniform-intrinsic-combine`, `amdgpu-enable-vopd`,
`amdgpu-internalize-symbols`, `amdgpu-ir-lower-kernel-arguments`,
`amdgpu-link-time-closed-world`, `amdgpu-load-store-vectorizer`,
`amdgpu-loop-prefetch` (**false**), `amdgpu-lower-global-ctor-dtor`,
`amdgpu-mode-register`, `amdgpu-opt-exec-mask-pre-ra`,
`amdgpu-opt-vgpr-liverange`, `amdgpu-reassign-regs`, `amdgpu-scalar-ir-passes`,
`amdgpu-scalarize-global-loads`, `amdgpu-sdwa-peephole`,
`amdgpu-set-wave-priority` (**false**), `amdgpu-simplify-libcall`,
`enable-amdgpu-aa`.

**Scheduler**: `amdgpu-disable-clustered-low-occupancy-reschedule`,
`amdgpu-disable-unclustered-high-rp-reschedule`,
`amdgpu-disable-rewrite-mfma-form-sched-stage` (**true**),
`amdgpu-schedule-relaxed-occupancy`, `amdgpu-use-amdgpu-trackers`,
`amdgpu-igrouplp-exact-solver`, `amdgpu-igrouplp-exact-solver-cost-heur`.

**Codegen prepare / lowering**: `amdgpu-bypass-slow-div`,
`amdgpu-codegenprepare-break-large-phis`,
`amdgpu-codegenprepare-force-break-large-phis`, `amdgpu-codegenprepare-mul24`,
`amdgpu-codegenprepare-expand-div64`,
`amdgpu-codegenprepare-disable-idiv-expansion`,
`amdgpu-codegenprepare-disable-fdiv-expansion`,
`amdgpu-codegenprepare-widen-constant-loads`,
`amdgpu-late-codegenprepare-widen-constant-loads`,
`amdgpu-use-divergent-register-indexing`, `amdgpu-disable-loop-alignment`,
`amdgpu-global-isel-new-legality`, `amdgpu-vgpr-index-mode`,
`amdgpu-use-aa-in-codegen`.

**Memory / LDS / alloca**: `disable-promote-alloca-to-lds`,
`disable-promote-alloca-to-vector`, `amdgpu-super-align-lds-globals`,
`amdgpu-asan-instrument-lds`, `amdgcn-skip-cache-invalidations`.

**Registers / spilling**: `amdgpu-spill-sgpr-to-vgpr`,
`amdgpu-spill-vgpr-to-agpr`, `amdgpu-spill-cfi-saved-regs`,
`amdgpu-prealloc-sgpr-spill-vgprs`, `amdgpu-mfma-vgpr-form`.

**Hazards / waitcnt**: `amdgpu-waitcnt-forcezero`,
`amdgpu-waitcnt-load-forcezero`, `amdgpu-expert-scheduling-mode`,
`amdgpu-wmma-vnop-hoisting`, `amdgpu-sgpr-hazard-boundary-cull`,
`amdgpu-sgpr-hazard-mem-wait-cull`.

**Next-use analysis presets**: `amdgpu-next-use-analysis-config` (string),
`-distance-cache`, `-count-phis`, `-forward-only`, `-precise-use-modeling`,
`-use-preheader-model`, plus three dump flags.

**Module splitting**: `-no-externalize-globals`,
`-no-externalize-on-addr-taken`, `-serial-execution`,
`-debug-proposal-search`, plus two output-path strings.

**Misc / test**: `amdgpu-xnack`, `amdgpu-sramecc`,
`amdgpu-fix-16-bit-physreg-copies`, `amdgpu-enable-merge-m0`,
`amdgpu-remove-redundant-endcf`, `amdgpu-stress-function-calls`,
`amdgpu-any-address-space-out-arguments`, `amdgpu-prelink`,
`amdgpu-use-native` (string), `amdgpu-enable-ocl-mangling-mismatch-workaround`,
`amdgpu-kernarg-preload`, `amdgpu-dump-hsa-metadata`,
`amdgpu-verify-hsa-metadata`, `amdgpu-print-rp-downward`,
`amdgpu-print-max-reg-pressure-regusage-{before,after}-scheduler`,
`r600-if-convert`, `r600-ir-structurize`, `amdgpu-function-calls`.

## 12. Per-function attribute knobs

The complete list of AMDGPU attributes that carry a **tunable quantity or
policy** (as opposed to ABI facts like `amdgpu-no-workitem-id-x`):

| Attribute | Read at | Default | Documented |
|---|---|---|---|
| `amdgpu-flat-work-group-size` | `AMDGPUSubtarget.cpp:164` | subtarget | yes |
| `amdgpu-waves-per-eu` | `AMDGPUSubtarget.cpp:225` | derived | yes |
| `amdgpu-num-sgpr` | `GCNSubtarget.cpp:544` | `MaxNumSGPRs` | yes (deprecated) |
| `amdgpu-num-vgpr` | `GCNSubtarget.cpp:624` | `Max` | yes (deprecated) |
| `amdgpu-agpr-alloc` | `GCNSubtarget.cpp:674` | `{~0u,~0u}` → half split | yes |
| `amdgpu-unroll-threshold` | `AMDGPUTargetTransformInfo.cpp:119` | **300** | yes |
| `amdgpu-promote-alloca-to-vector-max-regs` | `AMDGPUPromoteAlloca.cpp:359` | 32 | yes |
| `amdgpu-promote-alloca-to-vector-vgpr-ratio` | `AMDGPUPromoteAlloca.cpp:364` | 4 | yes |
| `amdgpu-max-memory-clause` | `SIFormMemoryClauses.cpp:275` | 15 | — |
| `amdgpu-hard-clause-length-limit` | `SIInsertHardClauses.cpp:199` | 255 | — |
| `amdgpu-max-memory-cluster-dwords` | `SIMachineFunctionInfo.cpp:184` | 8 | yes |
| `amdgpu-nsa-threshold` | `GCNSubtarget.cpp:891` | −1 → flag → 2 | — |
| `amdgpu-sgpr-hazard-mem-wait-cull-threshold` | `AMDGPUWaitSGPRHazards.cpp:569` | 8 | yes |
| `amdgpu-wave-priority-threshold` | `AMDGPUSetWavePriority.cpp:131` | 100 | yes |
| `amdgpu-sched-strategy` | `AMDGPUTargetMachine.cpp:603` | `""` | — |
| `amdgpu-post-sched-strategy` | `AMDGPUTargetMachine.cpp:626` | — | — |
| `amdgpu-post-ra-direction` | `GCNSubtarget.cpp:419` | — | yes |
| `amdgpu-max-num-workgroups` | `AMDGPULowerKernelAttributes.cpp:172` | — | yes |
| `amdgpu-dynamic-vgpr-block-size` | `Utils/AMDGPUBaseInfo.cpp:2472` | 0 | yes |
| `amdgpu-expert-scheduling-mode` | `SIInsertWaitcnts.cpp:3455` | flag | yes |
| `amdgpu-ieee` | `SIModeRegisterDefaults.cpp:19` | on | yes |
| `amdgpu-memory-bound` / `amdgpu-wave-limiter` | `AMDGPUMachineFunctionInfo.cpp:52,55` | set by PerfHint | — |
| `amdgpu-cooperative`, `amdgpu-tg-split`, `amdgpu-gds-size`, `amdgpu-dx10-clamp` | various | | partly |

**That is 20-odd per-function policy knobs against 159 module-wide flags.**
The gap is the finding.

---

# PART VI — Recommendations

## 13. Ranked promotion candidates

Criteria: (a) measurable performance effect, (b) plausibly kernel-dependent
optimum, (c) low implementation risk, (d) bounded search space.

### Tier A — high value, low risk. Do these first.

| # | Constant | Current | Proposed knob | Why |
|---|---|---|---|---|
| 1 | `ErrorMargin` | 3 | `amdgpu-sched-rp-error-margin` + attr | Directly costs 3 registers per region on every kernel. Kernels near an occupancy cliff can gain a whole wave. Search space 0-8. |
| 2 | `HighRPSGPRBias` / `HighRPVGPRBias` | 7 / 7 | `amdgpu-sched-high-rp-{sgpr,vgpr}-bias` | Controls how hard the high-RP stage fights pressure. No justification in source. |
| 3 | `MaxVGPRPressureInc` | 16 | `amdgpu-sched-max-vgpr-pressure-inc` | **Source already says "This is very inaccurate"** with two FIXMEs. Free win to expose. |
| 4 | `getInliningThresholdMultiplier()` | 11 | `amdgpu-inline-threshold-multiplier` + attr | Single most impactful IR-level policy. Cheap. Search space 1-20. |
| 5 | `NrOfSGPRUntilSpill` / `NrOfVGPRUntilSpill` | 26 / 32 | flags + attr | The only occupancy awareness in the AMDGPU inliner, and it ignores the kernel's actual wave target. |
| 6 | `amdgpu-unroll-threshold-*` | 2700/1000/200 | **already flags — add attributes** | Per-function `amdgpu-unroll-threshold` already exists; the private/local/if variants should follow suit. Mechanically trivial. |
| 7 | `amdgpu-memcpy-loop-unroll` | 16 | **already a flag — add attribute** | Comment names gfx1030 as its origin. Wrong by construction on other chips. |
| 8 | loop-alignment cutoffs | 64/128/192 | `amdgpu-loop-align-{max,cacheline}-size` | Instruction-fetch behaviour differs per generation; only GFX950 is special-cased today. |
| 8b | `SIFixSGPRCopies` score cutoff | 3 | `amdgpu-sgpr-to-valu-score-threshold` | Unweighted `Profit − Penalty` with a bare cutoff decides whether whole scalar chains move to VALU. No knob today; obvious candidate. |

### Tier B — high value, needs care.

| # | Constant | Current | Concern |
|---|---|---|---|
| 9 | `getMaxInterleaveFactor()` | 8 | Interacts with RP; a per-kernel value is genuinely right but changing it moves the whole vectorizer. |
| 10 | IGroupLP shape gate | `MFMAEnablement==2/4 && ExpRequirement==4 && TransPipeCount==32/64` | Should become a *predicate with tunable bounds*, not an equality test. Biggest single missed-optimisation surface for MFMA kernels. |
| 11 | `amdgpu-schedule-metric-bias` | 10 | Already a flag; **needs an attribute**. This is the natural top-level dial for a per-kernel latency-vs-occupancy tradeoff. |
| 12 | gfx90a AGPR half-split | `MaxVectorRegs / 2` | `amdgpu-agpr-alloc` already overrides it per function, so the work is to make the *default* smarter, not to add a knob. Source has a TODO. |
| 13 | `FenceLatency` | 2000 | A synthetic scheduling lever; exposing it is easy, but its interaction with `WriteBarrier=500` is unmodelled. |
| 14 | `SISchedule.td` `WriteVMEM`/`WriteBarrier` | 80 / 500 | Requires a subtarget-parameterised sched model, not a flag. Large but high-value. |

### Tier C — expose for experimentation only.

`getNumberOfRegisters() → 4` (#15) is the highest-leverage constant in TTI and
the most dangerous. Raising it will change vectorization and unrolling decisions
across the entire ecosystem and will cause spills on register-hungry kernels.
It should be exposed as a hidden flag for measurement **before** anyone proposes
a new default. Same for `getMinVectorRegisterBitWidth() = 32` and the 128-bit
load/store vector cap.

`SIMachineScheduler`'s 60/120/6/12 (#16) are only reachable via a non-default
scheduler, so tuning them has low blast radius but also low payoff.

`selectBITOP3`'s uniform-case threshold of 4 (`AMDGPUInstructionSelector.cpp:4476`)
is a hand-estimated SGPR↔VGPR move charge with no measurement behind it, on
gfx950+ only. Small blast radius, cheap to expose, and the divergent/uniform
pair makes it a clean natural experiment. Note the neighbouring `NumOpcodes == 2`
bail is *cosmetic* and should not be swept into the same flag.

### Not tuning, but worth fixing

`AMDGPULibCalls.cpp:985` — the `|c| <= 12` pow-expansion cutoff is justified by a
comment describing a linear multiply chain, but the code implements binary
exponentiation, so the real cost is `~2·log2(c)` and `pow(x,12)` is 4 fmuls, not
11. The threshold is mis-sized against its own stated rationale. This is a
standalone cleanup, independent of any autotuning work.

### Explicitly out of scope

- All `GCNHazardRecognizer` wait states, `SIInsertWaitcnts` limits,
  `AMDGPUInsertDelayAlu` maxima, allocation granules, `MaxAddressRegs`,
  `MaxRegisterSize` — **T0 correctness**.
- `amdgpu-max-memory-clause = 15` — physically bounded by counter width.
- `ScanLimit`, `SearchLimit`, `MaxUses`, `MaxIterationsCountToAnalyze`,
  `amdgpu-inline-max-bb`, `amdgpu-module-splitting-max-depth`, the src-modifier
  and canonicalization `MaxDepth`s (3/5), `GatherAllAliasesMaxDepth`,
  `MaxReorderWindow` — **compile-time budgets**, not performance knobs. Tuning
  them trades build time for quality and belongs to a different objective
  function.
- GlobalISel combiner `MaxIterations = 1` — real quality cost, but it is the
  convention across AArch64/RISCV/X86/WebAssembly/SPIR-V too. An LLVM-wide
  GlobalISel question, not an AMDGPU per-kernel knob.
- All R600 constants — deprecated VLIW target.

## 14. The knob pattern to use

Every new knob should follow the precedence already established by
`AMDGPUPromoteAllocaImpl::setFunctionLimits`, `GCNSubtarget::getNSAThreshold`,
and `SIFormMemoryClauses`:

```cpp
static cl::opt<unsigned> SchedErrorMargin(
    "amdgpu-sched-rp-error-margin", cl::Hidden, cl::init(3),
    cl::desc("Register pressure slack subtracted from scheduler RP limits"));

// In the consumer:
unsigned Margin = F.getFnAttributeAsParsedInteger(
    "amdgpu-sched-rp-error-margin", SchedErrorMargin);
if (SchedErrorMargin.getNumOccurrences())
  Margin = SchedErrorMargin;
```

Properties that make this the right pattern:

1. **The attribute is the per-kernel channel**; the flag is the global override.
2. **`getNumOccurrences()` gives the flag priority only when explicitly set**, so
   the flag never silently masks a deliberate per-function attribute.
3. **Zero behaviour change by default** — `cl::init` preserves today's constant,
   so the change is NFC and reviewable as such.
4. **The attribute survives IR round-trips**, so an autotuner can emit it from a
   front-end, a plugin pass, or `opt -passes=...` without touching the driver.
5. `AMDGPUAttributor` can compute and attach it, which is how a
   profile-guided or heuristic tuner would eventually plug in — exactly what
   `AMDGPUPerfHintAnalysis` already does for `amdgpu-memory-bound`.

Every promoted knob must also get a row in
`llvm/docs/AMDGPUUsage.rst` (the attribute table around lines 2497-2734) —
undocumented attributes are, in practice, unusable by external tuners.

**A note on precedence direction.** The backend actually uses *two* opposite
conventions, and picking the wrong one is a silent footgun:

- **Flag wins** (`getNumOccurrences()` pattern) — `AMDGPUPromoteAlloca`,
  `GCNSubtarget::getNSAThreshold`, `SIInsertHardClauses`,
  `AMDGPUNextUseAnalysis`. Rationale: the flag is a debugging override that must
  be able to defeat whatever the IR says.
- **Attribute wins** — `AMDGPU::getSchedStrategy` (`AMDGPUTargetMachine.cpp:602`):

  ```cpp
  StringRef llvm::AMDGPU::getSchedStrategy(const Function &F) {
    Attribute SchedStrategyAttr = F.getFnAttribute("amdgpu-sched-strategy");
    if (SchedStrategyAttr.isValid())
      return SchedStrategyAttr.getValueAsString();
    if (!AMDGPUSchedStrategy.empty())
      return AMDGPUSchedStrategy;
    return "";
  }
  ```

For autotuning, **flag-wins is the right default** — the tuner sets attributes,
and a human debugging a regression needs `-mllvm` to be able to override them
globally. Use attribute-wins only where the attribute encodes a semantic
requirement rather than a preference.

## 15. Autotuning strategy notes

**What makes AMDGPU harder than a CPU target:**

- **Occupancy is a step function.** VGPR budget is `alignDown(Addressable /
  Waves, Granule)`. Continuous knob changes produce discontinuous performance.
  Any search strategy must expect cliffs, which rules out pure gradient-like
  methods and favours evolutionary or Bayesian search with cliff-aware priors.
- **Register pressure couples everything.** Unroll threshold, interleave factor,
  inline multiplier, promote-alloca budget and scheduler RP limits all feed the
  same register file. They cannot be tuned independently; a joint search or a
  strict staging order is required.
- **The cost model is context-dependent** (§3.4), so a tuner that perturbs cost
  constants is perturbing a function whose value already depends on IR shape and
  fast-math flags. Fixing the FMA double-accounting is a prerequisite for
  trusting any cost-model tuning.
- **Module-wide flags cannot express per-kernel decisions.** Until the attribute
  channel is widened, "autotuning" in practice means recompiling the whole TU per
  configuration. This is the single change that unlocks everything else.

**Suggested phasing:**

1. **Phase 0 — instrumentation.** Expose Tier A constants as hidden flags only
   (NFC). Add attribute reads where the flag already exists (#6, #7, #11). Land
   as a series of small NFC patches.
2. **Phase 1 — flag sweep.** Random/grid search over the ~15 Tier A dimensions
   on a representative kernel suite. Objective: cycles, with occupancy and spill
   count as constraints. This produces the evidence needed to argue for new
   defaults.
3. **Phase 2 — attribute channel.** Promote the winners to per-function
   attributes, document them, and teach `AMDGPUAttributor` to set them from
   simple static heuristics (kernel size, MFMA density, LDS usage,
   `amdgpu-memory-bound`).
4. **Phase 3 — model.** Replace the static heuristics with an MLGO-style
   advisor. `AMDGPUPerfHintAnalysis` is the natural insertion point since it
   already emits per-function attributes consumed downstream.

See [`amdgpu-backend-autotuning.md`](amdgpu-backend-autotuning.md) for the
worked version of this plan applied to the CoExec scheduler and the MFMA rewrite
stage, including the specific finding that the CoExec strategy is currently
frequency-blind with **zero** exposed knobs.

---

## Appendix — reproducing the flag inventory

```bash
grep -rn "cl::opt<" --include=*.cpp --include=*.h llvm/lib/Target/AMDGPU \
  | grep -v "const cl::opt"     # 161 hits, 2 of which are
                                # isPassEnabled() parameter declarations
```

The three `AMDGPUTargetMachine.{cpp,h}` hits at `:180`, `:2762` and `.h:157` are
`bool isPassEnabled(const cl::opt<bool> &Opt, CodeGenOptLevel)` declarations, not
flag definitions. True flag count: **159**.

Note that `isPassEnabled` itself implements a third precedence rule worth
knowing about — a flag's default applies only above a minimum optimisation
level:

```cpp
bool AMDGPUCodeGenPassBuilder::isPassEnabled(const cl::opt<bool> &Opt,
                                             CodeGenOptLevel Level) const {
  if (Opt.getNumOccurrences())
    return Opt;
  if (TM.getOptLevel() < Level)
    return false;
  return Opt;
}
```

So several "default true" pass switches in the table above are in fact
**off at `-O0`/`-O1`** despite their `cl::init(true)`.
