# Auto-Tuning the AMDGPU Backend Codegen Path

Investigation of backend-path (post-ISel) auto-tuning for AMDGPU, focused on the
co-execution scheduler and the VGPR→AGPR MFMA-form rewrite, the "hot-cycle
detection lifts thresholds" thesis, PGO feasibility on GPU, and the
register-pressure / occupancy risk. All claims carry `file:line` anchors and
were cross-checked by an adversarial verification pass; corrections from that
pass are folded in.

---

## 0. TL;DR

- **Thesis verdict: sound, and half-built already.** Block-frequency (hot-cycle)
  weighting is a legitimate lever for both subsystems. `MachineBlockFrequencyInfo`
  is already wired into every scheduler stage, and the MFMA-rewrite stage
  *already* scales its cost model by block frequency. The coexec scheduler, by
  contrast, is completely frequency-blind — that's the clean greenfield.
- **The honest caveat: the frequency is *estimated*, not measured.** Typical
  amdgcn compiles carry no `branch_weights`/`entry_count`, so BFI comes from
  static branch heuristics + loop depth. Real device profiles need out-of-tree
  work (AMD's Offload-PGO-for-HIP RFC) that is **not in this tree**.
- **The register-pressure danger is real and is the crux.** GPU occupancy is a
  per-kernel step function of *peak* VGPR usage. Biasing scheduling toward hot
  loops can push peak pressure over a granule boundary and drop occupancy for the
  *whole* kernel. Every hot-cycle relaxation must be occupancy-guarded.
- **Recommended path:** (1) expose the load-bearing hardcoded constants as
  default-preserving knobs + build an external flag-sweep harness on a GEMM
  corpus; (2) upgrade the crude integer frequency weighting to the existing
  `FreqInfo` scheme and add occupancy-guarded hot-region relaxation; (3) optional
  MLGO advisor. Matrix/GEMM kernels are the ideal first target.

---

## 1. The AMDGPU backend tuning surface today

Tuning is entirely manual: ~43 `cl::Hidden` `-mllvm` flags, a handful of
per-function attributes, and frozen TableGen sched-model constants. **There is no
MLGO/autotuning harness wired to AMDGPU** — the only "advisor" string in the
target is a stale comment (`AMDGPURewriteAGPRCopyMFMA.cpp:312`).

**Cleanest entry point — per-function strategy selection.**
`GCNTargetMachine::createMachineScheduler` (`AMDGPUTargetMachine.cpp:1357-1385`)
resolves `getSchedStrategy(F)` (`:592`), which reads the **`amdgpu-sched-strategy`
function attribute first**, then the global `-amdgpu-sched-strategy` cl::opt.
Because it is a per-function attribute, a tuner can assign a different strategy
per kernel with no module-level recompile. Values: `max-ilp`,
`max-memory-clause`, `iterative-{ilp,minreg,maxocc}`, `coexec`, default
MaxOccupancy.

**Exposed scalar knobs (scheduler):** `-amdgpu-schedule-metric-bias` (10,
occupancy-vs-latency weight), `-amdgpu-vgpr-threshold-percent` (0, scales the
Excess/Critical RP limits), `-amdgpu-scheduler-pending-queue-limit` (256),
`-amdgpu-schedule-relaxed-occupancy`, `-amdgpu-use-amdgpu-trackers`,
`-amdgpu-disable-rewrite-mfma-form-sched-stage` (**true** — see §3).

**Occupancy budget:** `amdgpu-num-vgpr` / `-num-sgpr` / `-waves-per-eu` /
`-flat-work-group-size` attributes feed `getMaxNumVGPRs`/`getMaxNumSGPRs`
(`GCNSubtarget.cpp:531,611`). `-amdgpu-membound-threshold` (50),
`-amdgpu-limit-wave-threshold` (50).

**MFMA/matrix:** early form choice
`SIMachineFunctionInfo::selectAGPRFormMFMA` (`SIMachineFunctionInfo.h:1226`,
flag `-amdgpu-mfma-vgpr-form`, default false), IGLP GEMM interleave ratios
hardcoded (`AMDGPUIGroupLP.cpp:1024-1032` DS:2/MFMA:1;
`:2304-2324` VALU:2/DS_READ:4), `-amdgpu-mfma-padding-ratio` (0).
**No software pipeliner exists** — `SIInstrInfo` implements no
`analyzeLoopForPipelining`, so all cross-iteration MFMA overlap comes from
frontend unroll + IGLP.

**Frozen: the TableGen sched model.** All `ProcResource` counts, buffer sizes,
and latencies in `SISchedule.td` (`ReadAdvance<MIMFMARead,-4>` at `:223`, etc.)
are per-model constants whose own comments admit they are "Guessed" / "may not be
accurate". Tunable only by editing TableGen + rebuild; lowest feasibility as a
runtime knob but a high-value offline sweep target.

---

## 2. Focus A — Co-execution scheduler (`AMDGPUCoExecSchedStrategy.cpp`)

A strictly top-down, gfx1250-oriented pre-RA scheduler subclassing
`GCNSchedStrategy`. It runs on `GCNScheduleDAGMILive`, so **`DAG.MBFI` is
reachable** — the same handle the rewrite stage already uses. It models per-flavor
hardware-unit pressure via `HardwareUnitInfo` buckets keyed by
`InstructionFlavor` (WMMA, SingleCycleVALU, TRANS, MultiCycleVALU, VMEM, DS,
SALU, DMA, …), tracking busy `TotalCycles` and a `ProducesCoexecWindow` flag.

**It is 100% frequency-blind and has zero `cl::opt` — every threshold is a
literal or a comparator.** (Verified: grep for `cl::`/`opt<` over the whole
723-line file returns nothing.)

Decision sites a tuner or a hot-cycle signal would touch:

- **`tryEffectiveStall` (`:670-706`)** — the central overlap decision:
  `Costs.Effective = std::max({Ready, Structural, Latency})` (`:689`), then a
  strict `tryLess` (`:705`). Pure `max()`, no weights, no slack — a
  coexec-window producer that stalls even one cycle loses to a zero-stall filler.
  Highest-leverage single knob: replace with a weighted sum
  `(wReady,wStruct,wLat)` + a `StallSlack` tolerance, and scale the slack by
  region hotness.
- **`sortHWUIResources` (`:253-272`)** — the PR #169616 "ML-oriented selection"
  is really a hardcoded 4-key lexicographic comparator
  (`producesCoexecWindow > TotalCycles > size > flavor-enum`). This is exactly a
  linear score `a·window + b·cycles − c·size + d·flavorBias`; the coefficients
  become knobs or a learned table.
- **`ProducesCoexecWindow` flavor set (`:217-219`)** — `{WMMA, MultiCycleVALU,
  TRANS}`, set unconditionally. On gfx950/gfx1250 all VALU is single-cycle, so
  `MultiCycleVALU` is inert; worth gating behind a subtarget bitmask.
- **`getNextTargetSU` / LookDeep (`:219-228, 300-301`)** — an unbounded min-depth
  scan; a `MaxLookaheadDepth` cap is a natural knob.

**Two honesty caveats surfaced by verification:**

1. **Pre-RA, the `Structural` term is partially live (corrected by
   prototyping).** `getStructuralStallCycles` (`GCNSchedStrategy.cpp:277-312`)
   returns `max(` SchedModel reserved-resource stall `,` hazard-recognizer wait
   states `)`. The reserved-resource path (`:287-300`, `SU->hasReservedResource`)
   **is active pre-RA** and is nonzero for WMMA/MFMA, which reserve the matrix
   unit for multiple cycles — empirically `struct=15` for a
   `V_WMMA_SCALE_F32_16X16X128` in the coexec test. Only the hazard-recognizer
   path (`:302-309`) is post-RA-only (`!hasVRegLiveness`), so the sequence-
   dependent wait-state tables (`GCNHazardRecognizer.cpp:2223-2224`
   `WMMAWaitStates[]={5,9,3,5,9,17}`) are the part not visible pre-RA. Net: the
   structural stall lever **does** bite pre-RA via reserved resources — verified
   by the Phase-1 prototype, where `-amdgpu-coexec-stall-weight-struct=0` and a
   large `-amdgpu-coexec-stall-slack` each reorder the schedule. The earlier
   "Structural ≈ 0 pre-RA" claim was an overstatement.
2. **Likely bug in the sort comparator (`:266`):** the size tie-break uses
   `A.size() < B.size()` while the adjacent comment says "prefer more
   instructions." The code prefers *fewer*. This is incidental to auto-tuning but
   worth a separate fix/confirm — and it means the current comparator may already
   be doing the opposite of intent, which is itself an argument for making it a
   tunable/learned score.

**Selection is a soft guard, not hard:** `-amdgpu-sched-strategy=coexec` on a
non-gfx1250 target emits a `DS_Warning`
(`diagnoseUnsupportedCoExecSchedulerSelection`, `AMDGPUTargetMachine.cpp:604-612`)
but **still installs the coexec scheduler** (`:1379-1382` returns
`createGCNCoExecMachineScheduler` unconditionally). Useful for experimentation on
gfx942/gfx950.

---

## 3. Focus B — MFMA VGPR↔AGPR rewrite (`RewriteMFMAFormStage`)

A whole-function scheduling stage that speculatively reclassifies MFMA def/src2
registers from VGPR- to AGPR-form, prices the change, and keeps it only if
net-beneficial. Three-layer funnel:

**Layer 1 — eligibility (`initGCNSchedStage`, `~:1385-1396`):**
- arch gate `!ST.hasGFX90AInsts()` → skip;
- occupancy gate `MFI.getMinWavesPerEU() > 1` → skip (hardcoded 1: "AGPRs not
  used above occupancy 1");
- **primary trigger** `PressureBefore.getArchVGPRNum() >
  ST.getAddressableNumArchVGPRs()` (`:1392`) — fires **only for regions already
  spilling ArchVGPRs**. No hotness input at the trigger.

**Layer 2 — legality (`isRewriteCandidate` `:2352`, `isReachingDefAGPRForm`
`:2297`, `hasUseRequiringVGPR` `:2310`):** correctness filters (PR #200972 fixed
an illegal reclassification here). Not tuning targets.

**Layer 3 — cost model (`getRewriteCost` `:2472-2572`; decision `if (Cost > 0)
return false` at `:1409-1414`):**
- Spill cost `SpillCost = (SpillCostAfter − SpillCostBefore) * 2` (`:2516`) — the
  **×2 is the single most load-bearing hardcoded constant** in the accept
  decision (spill+restore). No knob.
- **All-or-nothing bail (`:2525-2528`):** if any evaluated excess-ArchVGPR
  region's freq-scaled `SpillCost > 0`, the *entire function* rewrite aborts.
  One cold region can veto rewrites for all hot regions.
- Copy cost `RC->getCopyCost() * (BlockFreq/EntryFreq)` per bridge
  (`:2543-2563`).

**It is already frequency-aware — but crudely (verified with a precision
correction).** `getRewriteCost` reads `DAG.MBFI` (`:2476`) and scales cost by
block frequency in two places, both lossy integer division but in *different*
ways:
- Spill scaling (`:2508-2522`): `RelativeFreq = larger/smaller of
  {EntryFreq, BlockFreq}`, always `≥ 1`. Its lossiness is a **dead-zone**: any
  block within ~2× of entry frequency collapses to `RelativeFreq = 1` (no
  scaling).
- Copy scaling (`:2545-2557`): `DefFreq/UseFreq = BlockFreq/EntryFreq`
  **truncates to 0** for any block colder than the entry — so cold-block copy
  costs vanish entirely. *(This is the real "quantize-to-0" site; the synthesis
  first mislocated it to the spill path — corrected here.)*

**Why it is disabled by default.** `DisableRewriteMFMAFormSchedStage` is
`cl::init(true)` (`:102-104`); the stage is only pushed into the pipeline when
the flag is flipped (`:786-787`). The "enable by default" change was reverted
(`ca5bc14df131`, #185604) because it "breaks a few tests / use cases
downstream." The coarse addressable-ArchVGPR trigger, the hardcoded ×2, the
all-or-nothing bail, and the crude integer frequency weighting are exactly the
fragility a hotness-aware, occupancy-guarded cost model would target.

*(Note: "no knob at the trigger" is literally true, but the whole stage is
knob-gated by `-amdgpu-disable-rewrite-mfma-form-sched-stage` and further gated
by the arch check and the cost model — the pressure test is the entry condition,
not the sole decision.)*

---

## 4. The hot-cycle thesis — rigorous verdict

**Sound, with a scope qualifier.** The mechanism is real and partly implemented:

1. **BFI is free everywhere in the scheduler.** `MachineSchedulerLegacy` /
   `PostMachineSchedulerLegacy` `addRequired<MachineBlockFrequencyInfoWrapperPass>`
   (`MachineScheduler.cpp:419,435,454`), threaded to `MachineSchedContext::MBFI`
   and captured by `ScheduleDAGMILive`, inherited as `GCNScheduleDAGMILive::DAG.MBFI`.
   Both coexec and the rewrite stage reach it with zero new pass wiring.
2. **The reuse template already exists.** `PreRARematStage::ScoredRemat::FreqInfo`
   (`GCNSchedStrategy.h:581-591`, ctor `:2987-3016`) constructs its *own*
   `MachineBlockFrequencyInfo(MF, MBPI, *DAG.MLI)` on the stack — self-contained,
   requiring no registered analysis — records per-region scaled uint64 freqs with
   a `ScaleFactor²` overflow guard, and drives `ScoredRemat::operator<`. **This is
   the correct model to copy:** it keeps fractional hotness instead of collapsing
   to 0/1 like the current rewrite cost model.
3. **Where it applies concretely:**
   - *Rewrite:* replace the crude integer `RelativeFreq`/`BlockFreq/EntryFreq`
     scaling (`:2508-2522, 2545-2557`) with the `FreqInfo` scheme; optionally add a
     hotness term to the trigger (`:1392`) so rewrites concentrate where cycles are
     spent, not merely where pressure exists; soften the all-or-nothing bail
     (`:2525`) into a freq-weighted global balance.
   - *Coexec:* compute per-region hotness = `MBFI->getBlockFreq(BB)/getEntryFreq()`
     once in `initialize()`/`initPolicy` (mirroring the pattern at
     `GCNSchedStrategy.cpp:2504-2512`), then raise `tryEffectiveStall` slack and
     relax the `RegMax` pressure comparison **in hot regions only**.

**The two limits to state honestly:**
- **"Lifting thresholds" is only as good as the BFI feeding it.** Static BFI is
  reasonable for *loop-nest structural* frequency but can be badly wrong on
  divergent GPU control flow (no `!prof` metadata; `getBlockProfileCount` returns
  `nullopt`). Mitigation: weight by loop-structural frequency, not intra-function
  branch skew. For GEMM this is sufficient (the inner loop dominates).
- **The pre-RA structural caveat (§2, corrected).** Pre-RA the Structural term
  is nonzero via SchedModel reserved resources (WMMA reserves the matrix unit),
  so the stall lever bites — but the sequence-dependent hazard wait states are
  still post-RA-only, so pre-RA tuning captures reserved-resource pressure, not
  the full hazard cost.

---

## 5. Register pressure and the occupancy cliff — the central risk

This is the user's own worry, and it checks out. Grounded in the real code:

**Occupancy is a step function of *peak* VGPR usage:**
```
waves = min( max( TotalNumVGPRs / alignTo(NumVGPRs, Granule), 1), MaxWaves )
```
(`AMDGPUBaseInfo.cpp:1469-1476`). For gfx90a/gfx942/gfx950
(`FeatureGFX90AInsts`): `Granule = 8`, `TotalNumVGPRs = 512`. So occupancy
= `512 / roundup8(peakVGPR)`, clamped. Crossing a granule-of-8 boundary that
changes that floor drops a whole wave:

| peak VGPR (after roundup8) | waves/EU |
|---|---|
| ≤ 64  | 8 |
| ≤ 72  | 7 |
| ≤ 80  | 6 |
| ≤ 96  | 5 |
| ≤ 128 | 4 |

**The cliff is per-kernel, not per-block.** `GCNRegPressure::getOccupancy`
(`GCNRegPressure.h:98-103`) takes the *max* pressure across the whole function.
So hot-loop-biased scheduling that raises peak VGPR in one region **lowers
occupancy for the entire kernel — including cold code that never needed the extra
registers.** A latency win in the hot loop can be erased by losing a wave
everywhere. This is precisely why naive "spend RP where it's hot" is dangerous on
GPUs.

**AGPR nuance (why the rewrite stage exists).** On gfx90a+ the register file is
*unified*: 512 total VGPRs shared by arch+acc, but ArchVGPRs are addressable-
capped at 256 (`getAddressableNumArchVGPRs`, `AMDGPUBaseInfo.cpp:1440-1445`). The
unified count is `alignTo(arch+avgpr, 4) + agpr` (`getUnifiedVGPRNum`,
`GCNRegPressure.h:74-82`). The rewrite fires on `getArchVGPRNum() > 256`
(`:1392`): ArchVGPRs *spill* while the acc half sits idle — moving MFMA operands
to AGPR relieves the arch sub-limit without growing the unified 512. So AGPR
rewrite is one of the few RP moves that can *raise* effective capacity rather than
just relocate pressure — which is exactly why it's worth doing right, and why the
hot-cycle weighting belongs in its cost model.

**`getVGPRSpills` (`GCNRegPressure.h:105-126`)** is the modeled-spill oracle the
rewrite stage already consults: `max(UnifiedSpill, ArchSpill + AGPRSpill)` against
three subtarget-derived thresholds. It is the natural hard veto for any hot-cycle
relaxation.

**Mandatory guardrails for any Phase-2 relaxation:**
1. **Occupancy-monotonicity guard** — never accept a hot-region relaxation that
   lowers `getOccupancy()` for the function. Check the *next granule boundary*,
   not just current pressure.
2. **Spill-count veto** — reuse `getVGPRSpills`: if a relaxation increases modeled
   spills in a hot block, reject (the rewrite stage's all-or-nothing bail is a
   blunt version of this; keep the spirit, soften the granularity).
3. **Loop-structural-only weighting** on divergent control flow.
4. **Default-preserving knobs** — every new `cl::opt` defaults to current
   behavior; the sweep is opt-in and cannot regress the default build.

---

## 6. PGO-on-GPU feasibility

- **Estimated BFI (today):** high feasibility, zero cost, but the numbers are
  guesses. Any profile-aware coexec/rewrite change built now runs on estimates
  and *auto-upgrades* to measured frequencies if real profiles ever land — no
  scheduler-side change needed.
- **Real device profiles (NOT in this tree):** the only concrete path is AMD's
  out-of-tree "Offload PGO for HIP" RFC — device instrumentation via
  `llvm.instrprof.increment`, wave-aggregated counters, host-side `.profraw`
  extraction, `-fprofile-use` → IR `branch_weights`/`entry_count` → MBFI
  reconstructs measured frequencies. Verified absent here
  (`amdgpu-flatten-spill-frequency`, `__profu_all` not present). API unsettled →
  **treat as a future upgrade, not a dependency.**
- **Sampling (rocprof / PC-sampling):** available, but there is **no AutoFDO
  ingestion path for amdgcn** (no `llvm-profgen` equivalent with correct amdgcn
  debug-line mapping). It can inform a human tuning knobs; it cannot be threaded
  into MBFI automatically. Low feasibility.

---

## 7. Are matrix/GEMM kernels the ideal first target? Yes.

- **Hot-cycle detection is nearly free and unambiguous** — a chained-MFMA GEMM
  has one inner loop that is trivially the hottest block; even estimated
  loop-depth BFI ranks it correctly, sidestepping the divergent-branch weakness.
- **The subsystems under study are matrix-specific** — RewriteMFMAForm exists for
  MFMA AGPR/VGPR form; coexec-window producers are WMMA/MFMA; IGLP GEMM strategies
  are the throughput lever. Matrix-only gating keeps the blast radius contained
  and may dissolve the default-off objection for RewriteMFMAForm.
- **Reward is runtime-measurable** — AGPR-vs-VGPR form, IGLP ratios, and padding
  all produce measurable occupancy/throughput deltas.
- *Caveat:* the hardcoded IGLP ratios may be tuned for a specific gfx942 shape and
  mis-serve gfx950 or attention (softmax/exp) loops. First target: **gfx942/gfx950
  GEMM**, attention as fast-follow.

---

## 8. Prior art / novelty

- **LLVM MLGO ships only two advisors** — inlining-for-size (`MLInlineAdvisor`)
  and regalloc eviction/priority (`MLRegAllocEvictAdvisor`). **There is no in-tree
  MachineScheduler advisor.** A coexec/rewrite advisor would be the *first
  scheduler advisor* and the *first on AMDGPU backend heuristics*.
- **Search-based autotuners** (OpenTuner, CompilerGym `llvm-ic-v0`, Google
  `ml-compiler-opt`) tune pass-ordering / `-mllvm` / `-O` flags at **module
  granularity**, not individual post-ISel heuristic decisions.
- **GPU kernel autotuners** (Triton `@triton.autotune`, rocMLIR/MIGraphX,
  TVM/Ansor) tune tile sizes / num_warps / num_stages / layouts at the
  **MLIR/frontend/template layer**, then hand finished IR to the backend. **None
  tune LLVM backend scheduler/regalloc heuristics** — confirming the premise that
  backend-path autotuning is unusual.
- **Closest prior art: CuAsmRL (CGO 2025)** — RL scheduling of NVIDIA **SASS**
  (post-assembly). This proposal differs by operating **inside the compiler on
  MIR** — portable, integrated, and profile-aware via already-wired BFI.
- **Predictable reviewer objection:** *"the heuristic is fine, just autotune the
  knobs."* That is exactly why Phase 1 (knob-sweep) comes first — it captures most
  of the gain cheaply and produces the evidence to justify anything heavier.

---

## 9. Phased plan

### Phase 1 — Expose constants + flag-sweep harness  (effort: small, risk: low)
Turn the load-bearing literals into default-preserving `cl::opt`s; sweep
externally on a GEMM/attention corpus.
- Coexec: `tryEffectiveStall` weights `(wReady,wStruct,wLat)` + `StallSlack`
  (`AMDGPUCoExecSchedStrategy.cpp:689,705`); `sortHWUIResources` coefficients
  (`:255-271`, and fix/confirm the `:266` size-comparator bug first);
  `ProducesCoexecWindow` bitmask (`:217-219`).
- Rewrite: spill multiplier `2` → `cl::opt<unsigned>` (`GCNSchedStrategy.cpp:2516`);
  percent-scale the ArchVGPR trigger (`:1392`, mirror `VGPRThresholdParser`
  `:108-120`).
- Sweep existing scalars (`-amdgpu-schedule-metric-bias`,
  `-amdgpu-vgpr-threshold-percent`, `-amdgpu-scheduler-pending-queue-limit`), IGLP
  ratios, `-amdgpu-mfma-vgpr-form`, and `amdgpu-sched-strategy`/`waves-per-eu`
  **as per-function attributes**.
- Harness: OpenTuner/CompilerGym-style wrapper around llc/clang measuring
  occupancy + runtime cycles. **Deliverables: proof of headroom + a labeled
  dataset for Phase 3.**

### Phase 2 — Hot-cycle-aware heuristics reusing `FreqInfo`  (effort: medium, risk: medium)
- *Rewrite:* replace crude integer frequency scaling (`:2508-2522, 2545-2557`)
  with the `FreqInfo` scheme (`:2987-3016`); soften the all-or-nothing bail
  (`:2525`) into a freq-weighted global balance. Re-baseline
  `sched_mfma_rewrite_cost.mir`.
- *Coexec:* per-region hotness in `initialize()`/`initPolicy`; scale
  `tryEffectiveStall` slack and relax `RegMax` pressure **in hot regions only**.
- **Guardrails from §5 are mandatory.** Correctness risk low (cost-model-only);
  tuning risk medium — needs GEMM perf validation. Works on estimated BFI today,
  auto-upgrades if Offload-PGO lands.

### Phase 3 — Optional MLGO advisor  (effort: large, risk: medium-high)
Factor the coexec tie-break and the RewriteMFMAForm skip behind an AMDGPU-local
advisor mirroring `RegAllocEvictionAdvisor`; features (occupancy, RP,
MFMA-consumer distance, BFI) as `TensorSpec`s; reward = stall-cycles/occupancy.
Reuse `llvm/lib/Analysis` `MLModelRunner` (`InteractiveModelRunner` for
experiments, Development/TFLite + `TrainingLogger` for RL). Train on the Phase 1
dataset. Do this **only if Phases 1–2 prove per-region decisions matter enough**
to justify the new MachineScheduler advisor plumbing (no in-tree precedent).

---

## 10. Incidental findings (worth separate fixes)
- **`sortHWUIResources` size tie-break (`AMDGPUCoExecSchedStrategy.cpp:266`)**
  uses `A.size() < B.size()` but the comment says "prefer more instructions" —
  the code prefers *fewer*. Confirm intent; possible latent bug.
- **Pre-RA the coexec picker sees reserved-resource stalls but not hazard wait
  states** (`GCNSchedStrategy.cpp:287-309`, hazard cost only when
  `!hasVRegLiveness`) — so WMMA matrix-unit occupancy is visible pre-RA but the
  sequence-dependent coexec wait-state tables are not. Worth a design note
  regardless of auto-tuning.
- **The ×2 spill multiplier (`:2516`)** may be a placeholder; sweeping it (Phase
  1) is the cheapest way to learn its true calibration.

---

## 11. Speculative / unverified — do not build on
- Offload-PGO-for-HIP is **not in this tree** and its final API is unsettled.
- The competitiveness of the `iterative-*` scheduler strategies is unassessed.
- `MultiCycleVALU` being inert on gfx950/gfx1250 needs confirmation before
  removing it from the coexec producer set.
- Whether Phase-1 per-kernel best flags generalize across problem shapes is the
  key empirical unknown the harness must answer.
