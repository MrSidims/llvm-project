# SLP `canConvertToFMA` — FMA fusion accounting

Status doc for the patch family around "don't let SLP vectorize a scalar
`fmul`+`fadd` chain when keeping it scalar fuses into cheaper FMAs".

Written 2026-08-15. Branch `slp-canconverttofma-standalone`, nothing pushed
beyond what is noted as merged.

---

## 1. The underlying problem

On targets with a fast FMA, a scalar chain `acc = acc + a[i]*b[i]` fuses into
one `v_fma` per element. If SLP vectorizes the `fmul` and the `fadd` into
separate vector ops, the fusion is lost and the result can be slower than the
scalar form it replaced. SLP has a helper that is supposed to notice this:

```cpp
static InstructionCost canConvertToFMA(ArrayRef<Value *> VL,
                                       const InstructionsState &S, ...);
```

It returns `FMACost` when fusing is cheaper than `fmul + fadd`, and
`InstructionCost::getInvalid()` when it is not. Two defects:

**Defect 1 — operand 0 only.** The helper inspected `Operands.front()` and
gave up if that was not an `fmul`. A left-associated accumulator chain always
puts the `fmul` on operand **1** (`fadd acc, (fmul a, b)`), so the shape the
check exists to protect was never recognised.

**Defect 2 — the fusion discount cancels both sides.** AMDGPU's TTI prices a
one-use contractable `fmul` feeding a contractable `fadd`/`fsub` as `TCC_Free`
(`AMDGPUTargetTransformInfo.cpp:585`), deliberately moving that cost onto the
`fadd`. The discount is gated on the context instruction `CxtI`, and
`TTI::getInstructionCost` forwards the instruction as `CxtI`
(`TargetTransformInfoImpl.h:1535`). So a per-instruction query returns
`fmul = 0`, and the comparison `fmul + fadd` vs `fmuladd` telescopes to
`0 + 1 - 1 = 0`. The penalty vanishes on exactly the targets whose fusion it is
meant to protect.

---

## 2. History

| What | Where | State |
|---|---|---|
| Precommit tests for ordered fadd-reduction FMA cost | #210835 | merged |
| Reduction penalty (`FusionSaving`) | #210399 → `bdedc49db2e5` | merged, then **reverted** |
| The revert | #216416 → `d6ea0866b832` | merged |
| `canConvertToFMA` operand selection + fmul costing | **#216425** | **open — this patch** |
| Re-land of the reduction penalty | #216426 | open |
| Precommit test for the operand-1 shape | #216428 → `47aa7b93b9cc` | merged |
| AMDGPU: sink one-use fmul to its fadd/fsub | #215810 | open |
| Issue: remove/replace the AMDGPU FMUL special case | #211092 | open |

### 2.1 Why #210399 was reverted

Not a functional bug and not a buildbot failure. alexey-bataev asked for the
revert on procedural grounds:

> Please revert
>
> — *"Is there anything wrong with the PR?"* —
>
> It was not approved, I told you what is expected from the patch

### 2.2 What "what is expected" means

This is the load-bearing part of the history, because it is the same request
that reappears as comment C4 on #216425. Across the #210399 discussion alexey
stated the condition three times, escalating each time:

> The change itself is correct, just need to pass types instead of actual
> instructions

> No, this is reinvention, use `canConvertToFMA`, but fix `canConvertToFMA` to
> use typed costs instead of actual instructions

> No, it is not ready, because it reinvents the check which already exist.
> Either land #211092 or change `canConvertToFMA` to use type-based cost
> estimation

So the agreed mechanism for dodging Defect 2 is **type-based costing inside
`canConvertToFMA`** — query `getArithmeticInstrCost(Opcode, Ty)` rather than
`getInstructionCost(I)`, because a type-based query carries no `CxtI` and
therefore cannot pick up the fusion discount.

Also from that thread, and already satisfied by the current patch:

- the `RK == Ordered && RdxKind == FAdd && RdxFMF.allowContract()` guard;
- extra tests for a multi-use `fmul` (fusion was never on the table, must not
  be penalised) and a no-`contract` chain (guard must bail).

---

## 3. The patch under review (#216425)

Commit `554fc8e30174`, plus a merge of `origin/main` (`2b59eba3b605`) and a
review-response commit (`6f88c8a36a58`). Net effect on
`llvm/lib/Transforms/Vectorize/SLPVectorizer.cpp`:

### 3.1 Look at both operands

```cpp
bool AllowReassoc = any_of(
    VL, [](Value *V) { return match(V, m_AllowReassoc(m_Value())); });
auto GetFMulOperandIdx = [&]() -> std::optional<unsigned> {
  for (unsigned Idx : seq<unsigned>(0, AllowReassoc ? 1 : Operands.size())) {
    InstructionsState CandS = getSameOpcode(Operands[Idx], TLI);
    if (!CandS.valid() || CandS.isAltShuffle() ||
        CandS.getOpcode() != Instruction::FMul)
      continue;
    if (!CheckForContractable(Operands[Idx]))  // pre-fix form, see §6.2
      continue;
    return Idx;
  }
  return std::nullopt;
};
```

The `AllowReassoc` gate restricts operand-1 detection to chains that do *not*
allow reassociation. A reassociative chain can be vectorized into a vector
`fmul` feeding a reduction, which is usually better than the scalar FMA chain
this check protects. The gate is also what keeps the patch inert on most
non-AMDGPU tests.

### 3.2 Price the fmul unfused

```cpp
auto GetUnfusedFMulCost = [&](Instruction *I) {
  assert(I->getOpcode() == Instruction::FMul && "Expected an fmul");
  TTI::OperandValueInfo Op1Info = TTI::getOperandInfo(I->getOperand(0));
  TTI::OperandValueInfo Op2Info = TTI::getOperandInfo(I->getOperand(1));
  return TTI.getArithmeticInstrCost(Instruction::FMul, I->getType(), CostKind,
                                    Op1Info, Op2Info,
                                    {I->getOperand(0), I->getOperand(1)});
};
```

No `CxtI` is passed, so Defect 2 cannot fire on this query. Note this is
*partially* type-based: the opcode and type are used, but operand value info is
still derived from the actual instruction. See §5, C4.

### 3.3 Fix the FMulAdd marking site (crash coupling)

```cpp
-      TreeEntry *FMulEntry = getOperandEntry(&E, 0);
+      TreeEntry *FMulEntry =
+          getOperandEntry(&E, IsOneUseVectorFMulOperand(LHS) ? 0 : 1);
```

**This must ship in the same commit as §3.1.** On `origin/main` the two sites
agree by accident: detection only ever finds the `fmul` at operand 0, and the
marking site hardcodes 0. Adding operand-1 detection without fixing the marking
site makes them disagree — the marking flips a *load* entry to
`CombinedVectorize` and SLP aborts with `Unhandled state`. Any split of this
patch that separates them produces a non-bisectable crashing commit.

---

## 4. Measurements

All on gfx90a, `llvm/test/Transforms/SLPVectorizer/AMDGPU/elementwise-fma-operand1.ll`,
kernel `axpy4_contract` = `d[i] = c[i] + a[i]*b[i]` for i in 0..3, `ptr noalias`,
`contract` on the FP ops. Build: Release + assertions, matching commit
`6f88c8a36a58`.

### 4.1 Tree cost, from `-debug-only=SLP`

```
unpatched:  store -3   fadd -2   load(c) -3   fmul -2   load(a) -3   load(b) -3   = -16
patched:    store -3   fadd -2   load(c) -3   [fmul absent]  load(a) -3   load(b) -3 = -14
```

The `fmul` bundle disappears from the patched tree because it is marked
`CombinedVectorize`, which removes its whole `VectorCost - ScalarCost` delta.

### 4.2 What that means, and what it does *not* mean

Vectorizing requires `Cost < -SLPCostThreshold`, so the vectorize/scalar flip
point moves from 16 to 14. **It does not mean the patch prefers scalar code.**
At any realistic threshold, including the default, both numbers say "vectorize"
and the emitted IR is identical.

Machine code for the two forms (verified with `llc`): both emit
3 × `flat_load_dwordx4` + 1 × `flat_store_dwordx4`. The difference is
**2 × `v_pk_fma_f32`** (vector) against **4 × `v_fma`** (scalar). So the true
saving from vectorizing here is 2 VALU.

That makes −14 the **more accurate** of the two numbers: it models a saving of
2, which is what actually happens. The unpatched −16 counts both the `fmul`
delta and the `fadd` delta, i.e. a saving of 4, pricing the chain as if no
fusion happened on the scalar side. Both remain optimistic in absolute terms,
because three load bundles contribute a phantom −9 — the scalar loads coalesce
into the same `dwordx4` loads either way. That phantom is a separate problem
and is not what this patch is about.

### 4.3 Threshold sweep — `axpy4_contract`, patched

| `-slp-threshold` | result |
|---|---|
| default (0) | `2 × <2 x float>` — identical to unpatched |
| 8 – 13 | `<4 x float>` |
| 14 – 15 | scalar |
| ≥ 16 | scalar |

Only 14 and 15 straddle the moved flip point, and the shipped test pins
`-slp-threshold=14`. That is the entire reason the test shows scalar IR.

### 4.4 Regression status

- Full SLP suite: 1066 passed / 24 unsupported / **0 failed**. Whole
  `llvm/test/Transforms`: 11688 discovered, **0 failures**.
- Default threshold, the two original kernels: baseline and patched produce
  **byte-identical output** (md5 `8b9dadbf…`), verified against a from-scratch
  `origin/main` build, not inferred from costs. **`axpy4_mixed_reassoc`, added
  later, is the exception** — it differs at the default threshold too, main
  `2 × <2 x float>` vs patched lane-0 scalar / `<2 x float>` / lane-3 scalar. So
  "NFC at the default configuration" holds over the *pre-existing* corpus only.
- gfx908 / gfx1030 / gfx1100 identical at every threshold 8–20; `dotconv`
  ordered reduction identical at every threshold 0–50 on gfx90a and gfx942.
- Net branch diff vs `origin/main` touches 2 files.

**Consequence: this patch is NFC at the default configuration, over the tested
corpus.** That is an empirical result across the SLP suite and the sweeps above,
not a proof of NFC. It regenerates no existing test — the branch diff vs
`origin/main` is the source file plus its own new test. Its only observable
effect in-tree is under an artificial threshold.

---

## 5. Review comments on #216425

| id | comment |
|---|---|
| 3787659951 | Precommit the test in separate NFC patch |
| 3789350834 | ` ```suggestion return match(V, m_AllowReassoc()); ``` ` |
| 3789352652 | Add assertion that I opcode is FMul |
| 3789353511 | Better not to do it, try to use type-based approach for all instructions |
| 3789354697 | Why do we prefer scalar form here? Vector is still more profitable |

### C1 — precommit the test

Resolved. Landed as #216428 / `47aa7b93b9cc`.

The `axpy4_mixed_reassoc` kernel added for C6 needs the same treatment, and it
qualifies — verified by round trip, not asserted:

| step | result |
|---|---|
| checks regenerated with an `origin/main` `opt` | lit green on `origin/main` |
| same file, patched `opt` | lit **fails** |
| regenerate with patched `opt` | **byte-identical** to the file at `7221528e02cf` |

Prepared as `slp-mixed-reassoc-precommit` @ `a32273016146`, one commit off
`origin/main`, 95 insertions in the test only. Rebase #216425 on it once it
lands, exactly as with #216428.

Which check blocks move, per kernel:

| kernel | CHECK, thr 14 | THR12, thr 12 | default thr |
|---|---|---|---|
| `axpy4_contract` | vector → scalar | unchanged | unchanged |
| `axpy4_reassoc` | unchanged | unchanged | unchanged |
| `axpy4_mixed_reassoc` | vector → scalar | vector → scalar | **also differs** |

The mixed kernel is the stronger regression test of the three, because it is the
only one that moves without a threshold flag, and the only one that moves at
both pinned thresholds. `axpy4_contract` sits on a cost knife-edge at
`Total Cost = -14`; the mixed kernel is declined earlier, by the seed deferral at
`:32745`, which never consults the threshold at all.

### C2 — `m_AllowReassoc()`

Right idea, but **the suggestion as literally written does not compile.**
`m_AllowReassoc` exists only as a template taking a sub-pattern
(`PatternMatch.h:96`); there is no nullary overload, and every in-tree use
passes one (e.g. `InstCombineMulDivRem.cpp:687`). The correct spelling is
`m_AllowReassoc(m_Value())`, which is exactly equivalent to the original
lambda: `AllowFmf_match::match` does the `dyn_cast<FPMathOperator>`, tests the
flag, then runs an always-true sub-match. Applied in `6f88c8a36a58`.

### C3 — assert the opcode is FMul

Correct, and stronger than it looks: at the call site the opcode is
**invariantly** `FMul`, so the `? :` that guarded it was dead code.

Why the invariant holds — for an `InstructionsState` whose main opcode is
`FMul`, any element with a different opcode is necessarily *copyable*, and
copyable elements already `continue` earlier in the loop:

- `isCopyableElement` returns true for anything non-binary
  (`SLPCompatibilityAnalysis.cpp:401`);
- its `BinOpSameOpcodeHelper` fallback cannot rescue a differing opcode,
  because `SupportedOp` (`SLPCompatibilityAnalysis.h:53`) does not contain
  `FMul`;
- `isExpandedBinOp` only ever maps `Add` ↔ `Shl x, 1`, so it does not apply to
  FP states.

Applied in `6f88c8a36a58`. Empirically confirmed: an instrumented build logging
every non-`FMul` instruction reaching that loop recorded **0 events across all
11688 tests in `llvm/test/Transforms`**. The assertion cannot fire and the
deleted arm was dead.

### C4 — "type-based approach for all instructions"

**Status: the fadd-side site is converted, measured NFC, and sitting unstaged in
the working tree next to the `CheckForContractable` fix.** Belongs in
**#216425** — see the measurement at the end of this section.

**This is the most consequential comment, and the follow-up commit only
half-addresses it.**

Read in isolation it looks like a nit about the ternary. Read against §2.2 it
is the same architectural condition alexey has now stated four times, and the
one that blocked and ultimately reverted #210399. He is not asking for the
`FMul` special-case to be deleted; he is asking for *every* cost query in
`canConvertToFMA` to be type-based.

The function currently mixes two conventions:

| site | current | type-based? |
|---|---|---|
| `GetUnfusedFMulCost` | `getArithmeticInstrCost(FMul, Ty, …, OperandValueInfo, Operands)` | partially — no `CxtI`, but still reads the instruction's operands |
| fadd side (VL loop) | `getInstructionCost(I)` | no — forwards `I` as `CxtI` |
| `FMACost` bail-out paths | `getInstructionCost(I)` / `getInstructionCost(OpI)` | no |

A fully type-based form would be roughly:

```cpp
FMulPlusFAddCost = NumOps * (TTI.getArithmeticInstrCost(S.getOpcode(), Ty, CostKind) +
                             TTI.getArithmeticInstrCost(Instruction::FMul, Ty, CostKind));
FMACost          = NumOps * TTI.getIntrinsicInstrCost(ICA, CostKind);
```

Notes before implementing:

- Dropping `CxtI` is the *point*, not a side effect — it is what makes the
  discount unreachable. Eight targets read `CxtI` in `getArithmeticInstrCost`
  (SystemZ, RISCV, Hexagon, AMDGPU, AArch64, ARM, PPC, X86), so this needs a
  full test run, not just AMDGPU.
- It also drops `OperandValueInfo`, so targets that price a constant operand
  cheaper will cost these `fmul`s slightly higher. That is consistent with the
  per-type form alexey accepted on #210399.
- **Trap:** the VL loop can contain *copyable elements* with non-binary
  opcodes. A blanket `getArithmeticInstrCost(I->getOpcode(), …)` there would
  read a non-existent operand 1. Price copyable elements as the state's main
  opcode (`S.getOpcode()`), which is also semantically what they are.

#### C4 — measured, and which patch it goes in

Applied to the fadd side, which is the one line #216425 left as unchanged
context:

```cpp
-    FMulPlusFAddCost += TTI.getInstructionCost(I, CostKind);
+    FMulPlusFAddCost +=
+        TTI.getArithmeticInstrCost(S.getOpcode(), I->getType(), CostKind);
```

| check | result |
|---|---|
| full `SLPVectorizer` suite | **1090/1090 pass**, zero check regenerated |
| threshold sweep, `-20…40` × 6 files × 4 targets | **byte-identical** to the pre-change build |
| is the line even live? | yes — poisoning it `×100` moves 22 sweep points in `X86/dot-product.ll` |

So it is NFC in practice on every in-tree case, including the eight `CxtI`-reading
targets. On AMDGPU it is NFC *by construction*: `AMDGPUTargetTransformInfo.cpp`
reads `CxtI` in `getArithmeticInstrCost` only under `case ISD::FMUL` (the fusion
discount) and under `ISD::FDIV` (`hasApproxFunc`). There is no fadd-inflation
path — the FMUL comment says the fmul is freed "supposing the fadd|fsub will get
estimated cost for the whole fused operation", but nothing implements that
supposition.

**Goes in #216425.** It is two lines, proven NFC, and it is the comment alexey
filed on that PR's diff — the same architectural condition that reverted #210399
(§2.2). Deferring a zero-risk NFC change to a follow-up leaves the stated
objection open on the PR it was made against. The one countervailing fact:
#216425 does not currently touch this line at all, so it is technically scope
expansion into unchanged context.

Not converted, deliberately: the two `FMACost += getInstructionCost(OpI/I)`
bail-out paths, which run when the operand is not a usable one-use fmul. Those
price the fadd alone with no fmul in play, so the discount cannot fire on them.

### C5 — "Why do we prefer scalar form here?"

He is right that the emitted vector code is better; he is wrong that the patch
chooses scalar. Nothing prefers scalar — the pinned `-slp-threshold=14` does.
See §4.2 and §4.3: the cost model still calls vector profitable (−14 < 0), and
at the default threshold the output is byte-identical to `origin/main`.

Measured `llc -mcpu=gfx90a` for the two forms, which is the answer to give him:

- vector: 3 × `flat_load_dwordx4` + **2 × `v_pk_fma_f32`** + 1 store
- scalar: 3 × `flat_load_dwordx4` (the backend re-merges the loads) +
  3 × `v_fma_f32` + 1 × `v_fmac_f32` + 1 store

So vector is better by 2 VALU — which is exactly the saving −14 models and −16
overstates.

The honest position is not to defend the test. Because the patch is NFC at the
default configuration (§4.4), a threshold-pinned cost probe is the *only* test that can
show it, and it will read as a regression to every reviewer who opens the file.

### C6 — `any_of` → `all_of`, and invert the reassoc polarity

Two independent changes in one suggestion. Measured separately, `ninja opt` +
the full SLP suite for each:

| variant | result |
| --- | --- |
| `any_of` + `AllowReassoc \|\|` (before) | 1090 / 1090 |
| `all_of` + `!AllowReassoc \|\|` (as suggested) | **5 fail** |
| `all_of` + `AllowReassoc \|\|` | 1090 / 1090 |
| `any_of` alone (loop unconditional, §6.6) | the same **5 fail** |

**`all_of` — taken.** NFC on the suite and it is the form the rest of
`canConvertToFMA` already uses: every other bundle-wide flag question in the
function is an `FMF &=` intersection, so "reassociative chain" should mean all
lanes, not one. With `any_of` a single reassoc lane disabled operand-1 search
for every lane.

NFC on the *existing* suite, but not NFC — the two spellings differ on a mixed
bundle, which nothing in tree covered. `axpy4_mixed_reassoc`, added in
`7221528e02cf`, is `d[i] = c[i] + a[i]*b[i]` x4 all `contract` with `reassoc` on
one lane's fadd only:

```
any_of:  VECTORIZED  -> fma path off, one lane vetoes the other three
all_of:  SCALAR      -> fma path on
```

It is red under `any_of` at `elementwise-fma-operand1.ll:177`. So `any_of` was a
latent bug, not a style question — the reduction path the gate protects needs
every link reassociable, so one lane cannot answer for the rest.

**The polarity — not taken.** Inverting it fails the same 5 tests that dropping
the gate entirely fails, for two different reasons, and the test FMF shows it
directly:

- The four X86 tests are `fast` / `reassoc` chains
  (`dot-product.ll`, `redux-feed-buildvector.ll`, `horizontal-fadd-with-sub.ll`,
  `select-logical-or-and-i1-vector.ll`). Under `!AllowReassoc` these newly
  search operand 1, get marked FMA-fusible, and lose the vector reduction the
  gate exists to protect.
- `AMDGPU/elementwise-fma-operand1.ll` — the patch's own regression test — is
  `contract` with no `reassoc`. Under `!AllowReassoc` it stops searching operand
  1, i.e. the feature the patch adds is switched off exactly where it is meant
  to fire.

The gate is not a correctness condition. Operand position never needs `reassoc`
— `a + b*c` puts the fmul on operand 1 with `contract` alone. It is
profitability, and it points the way it does because a reassociative chain has a
*better* alternative than the scalar fma chain this check protects.

#### C6 — what the 5 failures actually are

"5 tests fail" is not an argument about quality, so each was compiled both ways.
Counts are from `llc` on the SLP output, per function, using each test's own RUN
flags. Re-measured on `7221528e02cf` after a clean rebuild.

| test / function | current | inverted | |
| --- | --- | --- | --- |
| `select-logical-or-and-i1-vector.ll` `select_logical_and_i1` | 18 | 25 | regression |
| `select-logical-or-and-i1-vector.ll` `select_logical_or_i1` | 19 | 24 | regression |
| `redux-feed-buildvector.ll` `test` | 25 | 30 | regression |
| `dot-product.ll` `dot3f64_fast` | 7 | 8 | slight regression |
| `dot-product.ll`, other 11 functions | — | — | neutral, IR differs |
| `horizontal-fadd-with-sub.ll` `fsub_fmul_2` | 7 | **6** | **improvement** |
| `horizontal-fadd-with-sub.ll` `fsub_fmul_4` | 13 | 13 | wash |
| `elementwise-fma-operand1.ll` | 4 / 2 VALU | 2 / 4 | swap |

The two regressions are full-width vectorizations that collapse.
`redux-feed-buildvector` loses 2 × `<8 x double>` fmul + `vector.reduce.fadd`
for two serial 8-deep scalar `fadd` chains; `select-logical-or-and-i1` loses a
`<4 x float>` cmp/select/fmul/fadd and one vector store for 4 scalar lanes and 4
dword stores on avx512vl.

**The improvement is real and should not be papered over.**
`horizontal-fadd-with-sub.ll` `fsub_fmul_2` on znver4:

```
cur: vmulpd, vaddsd, vsubsd, vshufpd, vaddsd   = 5 arith
inv: vaddsd, vfnmadd231sd, vfmsub231sd         = 3 arith
```

`fsub_fmul_4` drops one arithmetic op the same way, 9 → 8, but pays it straight
back elsewhere and ends at 13 instructions either way, so `fsub_fmul_2` is the
only net win in the set. Both are short `reassoc` chains that only reach
`<2 x double>`, so the vector form pays a `vshufpd` extract per lane and gives up
the FMA — precisely the case this patch exists to catch. The gate blocks it
because `reassoc` is a *proxy* for "this chain vectorizes full width", and here
the proxy is wrong.

The AMDGPU row is not a quality question: the two kernels trade places, and
whichever takes the FMA path costs 4 VALU against 2. What matters is the
mechanism — under the inversion `axpy4_contract` no longer goes scalar at
threshold 14, i.e. operand-1 detection is off for plain-`contract` input, which
is `-ffp-contract=fast` without `-ffast-math` and the patch's whole target.

So the inversion trades two full-width vector wins for one narrow-vector FMA win
and disables the feature where it is meant to fire. Rejected — but the gate is
over-broad, not correct, and a width-aware predicate would get both. Separate
change, not this patch.

Unrelated overlap check: the follow-up branch moves `dot2f64_fast` /
`dot2f32_fast` in `dot-product.ll` in the *vectorizing* direction, a different
set of functions from the `dot3*` ones here. The two changes do not interact.

### C7 — don't rely on `Operands.size()`

Taken. `getNumberOfPotentiallyCommutativeOps(S.getMainOp())`. NFC today, since
`isAddSubLikeOp()` is asserted at entry so the main op is a binary
`Add`/`Sub`/`FAdd`/`FSub` and `buildOperands` returns exactly 2 columns for it.
It is the right bound anyway: the question the loop asks is "where can the fmul
sit", which is the commutative-operand count, not the column count.

Note it must be `getNumberOfPotentiallyCommutativeOps`, not `getNumOperands` —
and conversely `buildOperands` itself deliberately cannot use the helper
(SLPVectorizer.cpp:11916), because the helper collapses `fmuladd` to 2 while
`buildOperands` needs all 3.

Both applied in `f57929be7204`, suite 1090 / 1090.

---

## 6. Findings from an independent review

A second reviewer with no access to the reasoning above re-derived the patch
from scratch, building five variants of `opt` (baseline, patched, patch-minus-
marking-fix, patch-minus-reassoc-gate, instrumented). It reached the same
conclusions on C2–C5 and surfaced the following, which nobody on the PR raised.
Items marked **[verified]** were re-checked independently here.

**Its verdict: the approach is right — approve with changes.**

- Scanning both operands is the correct fix. `fadd` is commutative and
  `fadd acc, a*b` is the dominant shape, so the operand-0 assumption was simply
  a bug. A cleaner mechanism exists — ask the *tree's* operand entries, which
  SLP already commutatively normalises and which the marking site already uses,
  instead of re-deriving IR-source-order columns through `buildOperands`. That
  would give selection and marking one shared notion of which side holds the
  `fmul`, and would remove the need for the reassoc gate as a *correctness*
  device. It is a refactor, not a prerequisite.
- The `!AllowReassoc` gate is a profitability band-aid, not a legality
  condition. Keep it, but label it as a deliberate heuristic guarding reduction
  trees, because the real underlying issue is that forming the `fmuladd` can
  push a whole tree below threshold.
- Dropping `CxtI` is right and fixed at the right layer. The comparison being
  made is *unfused* scalar cost against fused cost, so the `fmul` must by
  construction be priced unfused. Passing the instruction asks TTI what the
  `fmul` costs *assuming it gets fused*, which is circular and collapses both
  sides of the comparison to the same number. This is a caller-semantics fix,
  not a target workaround — AMDGPU's discount is not itself wrong.
- **Blocking:** cover the marking hunk with a test, and mention that hunk in the
  PR description, which currently omits it entirely.
- **Strongly recommended in this same patch**, and they must go together —
  §6.7 shows that doing §6.2 alone converts a hidden wrong answer into a visible
  detection/marking disagreement: make `CheckForContractable` check the operand
  instructions' own FMF (§6.2), share the selected operand index between
  detection and marking (§6.7), and fix the surviving operand-0 assumption in
  the FMulAdd vector cost (§6.3). All three need a regression test; today none
  of them has one.
- **Done since this review:** the fadd-side cost loop no longer passes `CxtI`
  (the C4-wide item, §5 C4).
- **Follow-up:** the FSub asymmetry (§6.4) and documenting the seven call sites
  the broadened predicate feeds (§6.5).

### 6.1 The shipped test cannot catch the crash the patch fixes — **[verified]**

Building the patch *without* the §3.3 marking hunk and running the committed
test:

| `-slp-threshold` | result |
|---|---|
| 12 | **abort**, `Assertion 'E->State == TreeEntry::ScatterVectorize && "Unhandled state"' failed` |
| 13 | passes |
| 14 (the shipped RUN) | passes |

At threshold 14 the tree *is* built, costed, and marked — marking runs during
tree transformation and is threshold-independent. What threshold 14 changes is
that the tree is then rejected on cost, so the mis-marked entry never reaches
`vectorizeTree`, which is where the assertion lives. The most dangerous hunk in
the patch is therefore untested at 14, and that is why 12 is needed.

**But a `=12` RUN would be a crash guard only, not extra decision coverage.**
Measured baseline vs patched on the shipped test, gfx90a:

| `-slp-threshold` | baseline | patched | discriminates? |
|---|---|---|---|
| 10-13 | `<4 x float>` | `<4 x float>`, byte-identical | no |
| **14** | `<4 x float>` | scalar | **yes** |
| **15** | `<4 x float>` | scalar | **yes** |
| 16-18 | scalar | scalar, byte-identical | no |

Baseline cost is −16, patched is −14, and SLP vectorizes iff `Cost < -Threshold`,
so **both 14 and 15 discriminate** — an earlier version of this section said 14
was the only one, which was wrong. `=14` is the decision test; `=12` only
asserts that the two sites agree — worth
having, because that is the bug class that produced the layernorm abort, but it
must be labelled as a consistency guard rather than sold as a second cost check.

Also note the vectorized form at 12 is plain `fmul <4 x float>` +
`fadd <4 x float>`, not an `fmuladd` intrinsic. The `CombinedVectorize` marking
changes no emitted IR — it only removes the fmul entry's cost delta. In the
unfixed build the same marking lands on the *load* entry instead, which is what
aborts.

### 6.2 `CheckForContractable` does not actually check `contract` — **[verified, fix in the working tree]**

An `fmul` with **no** `contract` flag feeding a `contract` `fadd` is still
accepted and combined. Cost cliffs on gfx90a, patched vs baseline:

| input | baseline | patched |
|---|---|---|
| `fmul` **without** contract | vectorizes ≤15 | vectorizes ≤13 |
| `fmul` **with** contract | vectorizes ≤15 | vectorizes ≤13 |

Identical — so the FMA combine fired in both cases. Read-only explanation:
`CheckForContractable` (`SLPVectorizer.cpp:14362`) evaluates the *fadd's*
`InstructionsState S` and `continue`s on any element whose
`S.getMatchingMainOpOrAltOp(I)` is not S's main/alt op — which is every `fmul`.
`FMF` therefore stays `set()` and `allowContract()` is trivially true.

This is pre-existing (baseline called it on `Operands.front()`), but the patch
promotes it to the operand *selector*, so it now decides which operand is taken.
Consequence: SLP prices a fusion the backend will refuse — DAGCombine's
`isContractableFMUL` needs the `fmul`'s own flag — and under-prices the vector
form, biasing toward a scalar form that will not actually fuse. Fix: intersect
the FMF of the operand instructions directly rather than filtering them
through `S`. That is what the working tree now does — `CheckForContractable`
takes the bundle and reads each element's own flags, instead of routing them
through the fadd's `InstructionsState`.

Three in-tree tests move as a direct consequence, all of them intentional —
`X86/dot-product.ll`, `AArch64/wide-store.ll`, `AArch64/recalc-copyable-node.ll`.
Each had a non-contract `fmul` that was being priced as fusable. Regenerated.

### 6.3 The same operand-0 hardcoding survives elsewhere — **[fixed in the working tree]**

`getEntryCost`, `case TreeEntry::FMulAdd` (~`SLPVectorizer.cpp:17419`) does
`FMF &= cast<FPMathOperator>(FPCI->getOperand(0))->getFastMathFlags()`. With the
`fmul` on operand 1 this inspects the addend — usually a `load`, not an
`FPMathOperator` — so the fmul's flags are dropped from the `fmuladd` cost
attributes. Same bug class as Defect 1.

Now reads `FPCI->getOperand(E->CombinedFMulIdx)`, the index the selector chose
and the marking recorded, so it cannot drift from either.

### 6.4 FSub acceptance is dead in the transform — **[fixed in the working tree]**

`canConvertToFMA` returned a valid cost for `fsub c, a*b`, but `transformNodes`
`break`s for FSub unless the `fmul` is the LHS, so it was never combined. Tree
costs stayed consistent, but the acceptance still fed the six other consumers in
§6.5. Resolved the conservative way — the selector now stops at operand 0 for
FSub, matching what the marking will accept:

```cpp
bool OnlyFirstOperand = AllowReassoc || S.getOpcode() == Instruction::FSub;
```

Extending the marking instead is the other option, since the backend does fold
`fsub x, (fmul y,z)` into `fma (fneg y), z, x`, but that is a new capability
rather than a consistency fix. Zero test movement either way.

### 6.5 The patch broadens seven call sites, not just the cost model — [verified]

`canConvertToFMA` is called from seven places — `:13638` (`getNumScalarInsts`),
`:15251` (the FMulAdd marking), `:16969` (`GetFMulAddCost`), `:31839` and
`:31939` (reductions), `:32745` (**seed deferral**, where fadd/fsub that "could
be an FMA" are pushed to a retry pass), and `:34020`. Accepting operand 1 changes
seed ordering, not only cost. In-tree fallout is zero, but the PR description
should say so.

### 6.6 The reassoc gate is load-bearing — **[verified]**

Making the operand loop unconditional breaks **5** SLP tests:
`X86/dot-product.ll`, `X86/horizontal-fadd-with-sub.ll`,
`X86/redux-feed-buildvector.ll`, `X86/select-logical-or-and-i1-vector.ll`, and
the new AMDGPU one. The gate is justified — but that justification belongs in
the commit message. It was `any_of` over the bundle, so one reassoc lane
disabled operand-1 search for every lane; now `all_of` per C6.

### 6.7 The checker fix exposes a second detection/marking split — **[fixed in the working tree]**

Detection selects the fmul index by *contractability*
(`SLPVectorizer.cpp:14393`); marking selects it by *shape*,
`IsOneUseVectorFMulOperand(LHS) ? 0 : 1` (`:15257`). When both operands are
one-use fmuls and only the second is contractable, they disagree.

Repro, gfx90a, `-slp-threshold=-100`, `!AllowReassoc`, per lane
`%m = fmul float %a, %b` (no contract), `%n = fmul contract float %c, %d`,
`%s = fadd contract float %m, %n`:

```
SLP: Skipping cost for combined node that starts with   %m0 = fmul float %a0, %b0.
```

Detection selected `%n` (operand 1, contractable); marking discounted `%m`
(operand 0, **not** contractable). The cost model charges the fmul that will
actually fuse and forgives the one that will not — wrong in both directions.

Severity is **cost-only**, not correctness: `TreeEntry::FMulAdd` has no handler
in `vectorizeTree`, so nothing about the emitted IR changes. Grepping `FMulAdd`
finds only the enum (`:3065`), the marking (`:15255`), and two cost sites
(`:17409`, `:17587`).

Attribution matters here. `origin/main` marks the *same* node, so this is not a
regression the operand-1 detection introduces — on `origin/main` and on #216425
as committed, detection wrongly *accepts* the non-contract operand-0 fmul (§6.2),
so the two sites agree for the wrong reason. The `CheckForContractable` fix makes
detection correct, and that is what turns a hidden wrong answer into a visible
disagreement. Fixing §6.2 without sharing the index trades one bug for another.

**Remedy — implemented in the working tree.** One selector produces the index and
every consumer reads it. `canConvertToFMA` gained an out-param, marking uses it
instead of the shape test, the marked entry is re-checked for foldability, and
§6.3's FMF intersection reads the same index:

```cpp
unsigned FMulIdx = 0;
if (!canConvertToFMA(E.Scalars, E.getOperations(), *DT, *DL, *TTI, *TLI,
                     &FMulIdx).isValid())
  break;
TreeEntry *FMulEntry = getOperandEntry(&E, FMulIdx);
if (!IsOneUseVectorFMulOperand(FMulEntry))
  break;
E.CombinedOp = TreeEntry::FMulAdd;
E.CombinedFMulIdx = FMulIdx;
```

Verified: the repro now discounts `%n0 = fmul contract` instead of `%m0 = fmul`.

**How observable is the disagreement itself?** Barely. The discount removes an
`fmul` bundle's `VectorCost - ScalarCost`, and both candidate bundles are the
same opcode, type and width, so the totals match whichever one is picked. Swept
thresholds −20…40 on gfx90a for four shapes — plain two-fmul, constant on the
RHS fmul, constant on the LHS fmul, and `a*a` on the LHS — and **none**
discriminate. So the disagreement is a real reasoning error with no cost
consequence in any shape that could be constructed.

The shape that *is* observable is the new foldability re-check. With operand 0 a
non-contract one-use fmul and operand 1 a **multi-use** contract fmul, the
selector picks operand 1, which cannot be folded, so nothing is discounted and
the entry keeps its negative delta:

| `-slp-threshold` | `origin/main` | #216425 | with these fixes |
|---|---|---|---|
| 15 | vector | vector | vector |
| **16, 17** | scalar | scalar | **vector** |
| 18+ | scalar | scalar | scalar |

Vectorizing is the honest answer there — the multi-use fmul's result is stored,
so it will not fuse, so charging for it is correct.

---

## 7. Open decisions

1. ~~**C4, narrow or wide.**~~ **Resolved — wide, in #216425.** The fadd-side
   loop at `SLPVectorizer.cpp:14431` is converted; measured NFC across the whole
   suite and a 61-point threshold sweep. See C4's measurement subsection. It
   currently sits unstaged, so it is not yet bound to a commit.

2. **What to do with the test.** Keep it, and keep `=14` — measurement in §6.1
   shows it is the *only* threshold at which the patch is observable, so it is
   the decision test the file was written to be. **Done:** the `=12` RUN is
   staged as a consistency guard for the marking hunk, and the header comment is
   reworded — its old phrasing, not the cost model, is what drew comment C5. The
   new header still overstates what `=12` covers, see §6.7.

3. **`elementwise-fma-operand1.ll` is already upstream** via #216428, so the PR
   should rebase to show only the CHECK-line delta.

4. **#211092.** alexey offered it as the alternative to type-based costing
   ("Either land #211092 or …"). Still open; arsenm redirected it toward fixing
   at the `fadd` use and reviving `isProfitableToSinkOperands`, which is what
   #215810 does.

---

## 8. The split

The checker fix and everything it drags in were **moved off #216425** into a
follow-up branch. The trigger was that `815d7097103b "extra fixes"` had landed
the checker fix on the PR branch while the three CHECK updates it forces were
still uncommitted, so the branch as pushed would have been red.

**`slp-canconverttofma-standalone`** — #216425, now the narrow patch again:

| commit | what |
|---|---|
| `554fc8e30174` … `6f88c8a36a58` | the original patch plus review fixes |
| `815d7097103b` | `m_AllowReassoc(m_Value())`, comment reword, C4 type-based fadd cost, threshold-12 guard |
| `6856282c74e0` | takes the checker fix back out |
| `2d9bb34d2f3a` | FSub stops at operand 0, §6.4 |

Two files vs `origin/main`, SLP suite **1090 / 0 failed**, no existing test
regenerated. Each commit builds and passes on its own.

**`fix-slp-canconverttofma-contract-check`** — the follow-up, one commit
`790aaa85a62b`. It is the coherent unit §6.2, §6.3 and §6.7 all point at, and it
cannot be split further, because fixing the checker is exactly what makes
detection and marking disagree.

| file | what |
|---|---|
| `SLPVectorizer.cpp` | `CheckForContractable` reads each element's own FMF; `canConvertToFMA` publishes the chosen index via `SelectedFMulIdx`; `TreeEntry::CombinedFMulIdx` carries it to marking and to the FMulAdd vector cost; marking re-checks foldability |
| `AMDGPU/fma-operand-contract-selection.ll` | new, 3 kernels on gfx90a at two thresholds |
| `X86/dot-product.ll`, `AArch64/wide-store.ll`, `AArch64/recalc-copyable-node.ll` | regenerated |

SLP suite **1091 / 0 failed**. Both RUN lines of the new test fail on the parent
branch tip, one per defect.

### 8.1 What the new test pins

| kernel | discriminates at | catches |
|---|---|---|
| `nocontract_mul_operand1` | `-slp-threshold=15` | §6.2 — a non-contract operand-0 fmul being accepted |
| `contract_mul_operand1` | — | control, the same shape *with* `contract`, must stay vector |
| `selected_mul_multiuse` | `-slp-threshold=16` | §6.7 — marking discounting an fmul that cannot be folded |

Trimmed from the first draft. `two_muls_only_operand1_contract`, the canonical
§6.7 disagreement shape, was dropped — it produces identical output on buggy and
fixed builds, so it froze behaviour without proving anything. The gfx942 and
gfx950 RUN lines went too — all three CDNA targets emit byte-identical IR at
both thresholds, modulo the `target-cpu` attribute string.

### 8.2 The regenerated tests are behaviour changes, not bookkeeping

This is the follow-up's real review burden and it should not be presented as a
mechanical update.

- `AArch64/recalc-copyable-node.ll` `test1` goes from a fully vectorized
  `<2 x double>` / `<4 x double>` chain to **entirely scalar** — roughly 30
  vector ops replaced by 25 scalar ones.
- `AArch64/wide-store.ll` goes from one `<4 x float>` fadd to a scalar store, a
  `<2 x float>` fmul/fadd pair, and another scalar store.
- `X86/dot-product.ll` moves only under AVX2.

The direction is the tell. A *stricter* checker marks fewer FMulAdd nodes, which
keeps the fmul bundle's negative delta in the total and should bias toward
**more** vectorization. These move the other way, so the cost model is not what
is driving them — the seed-deferral caller at `:32745` is. Declining to defer a
seed changes which trees get built at all, which is exactly the §6.5 concern
with a concrete instance under it. Nobody has yet argued these three are
improvements.

---

## 9. Code map

| What | Where |
|---|---|
| `canConvertToFMA` | `SLPVectorizer.cpp` ~14350 |
| FMulAdd marking site | `SLPVectorizer.cpp` ~15271 |
| AMDGPU fmul fusion discount | `AMDGPUTargetTransformInfo.cpp:585` |
| `getInstructionCost` → `CxtI` forwarding | `TargetTransformInfoImpl.h:1535` |
| `m_AllowReassoc` | `PatternMatch.h:96` |
| `isCopyableElement` | `SLPCompatibilityAnalysis.cpp:385` |
| `BinOpSameOpcodeHelper::SupportedOp` | `SLPCompatibilityAnalysis.h:53` |
| `isExpandedBinOp` | `SLPCompatibilityAnalysis.cpp:441` |

Build: `/home/mrsidims/LLVM-work/build` (Release + assertions).
Suite: `bin/llvm-lit -q llvm/test/Transforms/SLPVectorizer/`.
