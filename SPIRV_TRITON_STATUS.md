# Triton/Gluon SPIR-V Pipeline Status

**Last updated:** 2026-04-11
**Active branch:** `triton-compatible-spirv` at `/home/mrsidims/LLVM/llvm-project`
**Base:** `7f77ca0dbda4` (Triton 3.3.1's expected LLVM version)

## Where we are

We have all LLVM + Triton source-level changes in place. The only remaining blocker is rebuilding LLVM with **both LLVM and MLIR as static-only** (no shared libs) to avoid double-registration of command-line options when Triton's `libtriton.so` loads.

## Branches and commits

### LLVM: two parallel branches

1. **`full-triton-intrinsics`** (at `/home/mrsidims/LLVM/llvm-project`) — original development branch with all patches, tests passing (`check-llvm` clean). Based on a commit ~20 upstream commits newer than `7f77ca0dbda4`. **NOT Triton-compatible** due to MLIR API changes (`OpaqueProperties` signature on `inferReturnTypes`).

2. **`triton-compatible-spirv`** (current) — same patches cherry-picked onto `7f77ca0dbda4`. Cherry-picked commits (10 total, in order):
   - `9aecd268a6bd` — `[SelectionDAG] Add expansion for llvm.convert.to.arbitrary.fp`
   - `2c58f3a989f8` — wip (from `1fc4d8bbb417`) — needed include conflict resolution in `SPIRVEmitIntrinsics.cpp`
   - `22ee8a078f4e` — wip (from `688119cc55d4`)
   - `6a895c2e4b70` — wip (from `d39516b79b91`)
   - `60d6f4e7e324` — `wip on coop matrices` (from `b11b8de7f4ba`) — needed conflict resolution in `IntrinsicsSPIRV.td` and `SPIRVInstructionSelector.cpp` (both accept-incoming for the new `spv_named_boolean_spec_constant` / `spv_spec_constant` intrinsics)
   - `e88e32946add` — `[AMDGPU][test] Update llc-pipeline manifests for coop matrix pass`
   - `40dc94ca80ce` — `[AMDGPU][coopmatrix] Fix critical correctness bugs`
   - `bb7c046edebd` — `[AMDGPU][coopmatrix] Add FP8 MFMA, larger f16/bf16, and f64 entries`
   - `2b234c4937a7` — `[AMDGPU][SPIRV][coopmatrix] Implement tasks 8, 10, 11, 12`
   - `78c64814cff3` — wip (coopmatrix type interpretation for FP8 + SPIRVPrepareFunctions fix + Triton DotOp updates)

**Post-cherry-pick fix applied:**
- `llvm/lib/Target/SPIRV/SPIRVEmitIntrinsics.cpp`: `TM.getSubtarget<SPIRVSubtarget>(F)` → `TM->getSubtarget<SPIRVSubtarget>(F)` (older LLVM had `TM` as pointer).

### Triton working tree (`/home/mrsidims/Triton/triton`)

Has uncommitted changes for the full SPIR-V path:
- `python/triton/knobs.py` — adds `TRITON_AMD_EMIT_SPIRV` + `TRITON_AMD_USE_GENERIC_FP_CONVERT` knobs
- `third_party/amd/backend/compiler.py` — `make_spirv()` stage, `spirv64-amd-amdhsa` triple, `amdgpu_kernel` → `spir_kernel` CC rewrite
- `third_party/amd/lib/TritonAMDGPUToLLVM/TritonGPUToLLVM.cpp` — SPIR-V target detection + pattern gating
- `third_party/amd/lib/TritonAMDGPUToLLVM/SPIRVTargetInfo.{h,cpp}` — SPIR-V TargetInfo (shuffles, ballot, barriers, programId)
- `third_party/amd/lib/TritonAMDGPUToLLVM/ElementwiseOpToLLVM.cpp` — `GenericFpToFpOpConversion` covering all 26 FP8 paths (SW only, no ROCDL) + `populateGenericElementwiseOpToLLVMPatterns`
- `third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM.cpp` — `SPIRVDotOpConversion` + `SPIRVScaledDotOpConversion`
- `third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/CoopMatrix.cpp` — `convertCoopMatrix()` emits `llvm.coopmatrix.muladd` with type-interpretation i32 args (APFloat::Semantics enum values for FP8)
- `third_party/amd/lib/TritonAMDGPUToLLVM/PatternTritonGPUOpToLLVM.h` — new function declarations
- `CMakeLists.txt` + `third_party/amd/lib/TritonAMDGPUToLLVM/CMakeLists.txt` — add `LLVMSPIRVCodeGen` and new source files

**Post-build fix:** `getTypeInterpretation()` in `CoopMatrix.cpp` was using `llvm::TypeSwitch` which requires `#include "llvm/ADT/TypeSwitch.h"`. Replaced with a simple `isa<>` chain — compiles cleanly.

## The blocker: LLVM shared lib double-registration

When `libtriton.so` (Python C extension) is loaded:
```
CommandLine Error: Option 'print-pipeline-passes' registered more than once!
LLVM ERROR: inconsistency in registered CommandLine options
```

**Root cause:** `/home/mrsidims/LLVM/build` is configured with `LLVM_BUILD_LLVM_DYLIB=ON` + `LLVM_LINK_LLVM_DYLIB=ON`. Triton's `libtriton.so` links against static LLVM `.a` archives (`LLVMAMDGPUCodeGen`, `LLVMSPIRVCodeGen`, etc. per `CMakeLists.txt:360-366`), but `libLLVM.so.23.0git` also gets loaded at runtime via RPATH → same cl::opt registered twice.

**Fix in progress (build was killed here):** reconfigure LLVM with:
```
-DLLVM_BUILD_LLVM_DYLIB=OFF
-DLLVM_LINK_LLVM_DYLIB=OFF
-DCLANG_LINK_CLANG_DYLIB=OFF
-DMLIR_BUILD_MLIR_DYLIB=OFF
-DMLIR_LINK_MLIR_DYLIB=OFF
```

Then `ninja -j8` to rebuild (kill was at ~4940/7416 targets — about 66% through). After LLVM rebuilds, rebuild Triton's `libtriton.so` in the existing `/home/mrsidims/Triton/triton/build/cmake.linux-x86_64-cpython-3.10` dir.

## Next steps when resuming

1. **Reapply the cmake config** (flags above) in `/home/mrsidims/LLVM/build`:
   ```bash
   cd /home/mrsidims/LLVM/build && \
   cmake -DLLVM_BUILD_LLVM_DYLIB=OFF -DLLVM_LINK_LLVM_DYLIB=OFF \
         -DCLANG_LINK_CLANG_DYLIB=OFF -DMLIR_BUILD_MLIR_DYLIB=OFF \
         -DMLIR_LINK_MLIR_DYLIB=OFF /home/mrsidims/LLVM/llvm-project/llvm
   ```
2. **Rebuild LLVM+MLIR+LLD** with `-j8` (NOT `-j16` — WSL OOMs on some files):
   ```bash
   ninja -j8
   ```
3. **Delete stale Triton cmake cache** (it has outdated LLVM library references):
   ```bash
   rm -rf /home/mrsidims/Triton/triton/build
   rm -f /home/mrsidims/.triton/llvm/llvm-ubuntu-x64
   ```
4. **Rebuild Triton** (already-compiled source; only relinks `libtriton.so`):
   ```bash
   cd /home/mrsidims/Triton/triton && \
   LLVM_SYSPATH=/home/mrsidims/LLVM/build TRITON_HOME=/home/mrsidims/.triton \
   python3 setup.py bdist_wheel
   ```
5. **Copy `libtriton.so` to source tree** (editable import layout):
   ```bash
   cp /home/mrsidims/Triton/triton/build/lib.linux-x86_64-3.10/triton/_C/libtriton.so \
      /home/mrsidims/Triton/triton/python/triton/_C/libtriton.so
   rm -f /home/mrsidims/Triton/triton/python/triton/_C/libtriton/libtriton.so
   ```
6. **Test import:**
   ```bash
   pip uninstall triton -y
   PYTHONPATH=/home/mrsidims/Triton/triton/python python3 -c \
     "import triton; from triton import knobs; print('emit_spirv:', hasattr(knobs.amd, 'emit_spirv'))"
   ```
7. **Run E2E vector_add with `TRITON_AMD_EMIT_SPIRV=1`** — see `/tmp/test_spirv_compile.py` for the compile harness skeleton (needs updating for the current ASTSource API that uses `constexprs={"BLOCK": 1024}` and `signature={"a_ptr": "*fp32", ...}`).

## What already works (verified before the build blocker)

Tested via direct `llc` invocation on hand-written LLVM IR (build dir pre-static-rebuild):

| Test | Triple | Result |
|---|---|---|
| Vector add (load/add/store f32, AS 1) | `spirv64-amd-amdhsa` | PASS |
| SPIR-V builtins (`__spirv_GroupNonUniformShuffle`, `_Ballot`, barrier) | `spirv64-amd-amdhsa` | PASS |
| Shared memory (AS 3 → Workgroup) + fptrunc/fpext | `spirv64-amd-amdhsa` | PASS |
| Coop matrix `muladd` (f16×f16→f32) with 11-arg type-interp signature | `spirv64-amd-amdhsa` | PASS — emits `OpCooperativeMatrixLoadKHR` / `OpCooperativeMatrixMulAddKHR` / `OpCooperativeMatrixStoreKHR` |

Key fix verified working: `SPIRVPrepareFunctions.cpp` default-case switch now has explicit `Intrinsic::coopmatrix_*` passthrough cases (preventing the AMD-vendor path from wrapping them into `@spirv.llvm_coopmatrix_*` external calls).

## check-llvm status

`ninja check-llvm` on `full-triton-intrinsics` (pre-cherry-pick): **0 failures** after updating the 2 numerical coop-matrix tests (removed explicit `declare`s, let LLVM auto-generate from the updated intrinsic signature). Same patches are now on `triton-compatible-spirv` — need to re-run `check-llvm` there once the static rebuild is done.

## Gaps NOT yet addressed (for after E2E pipeline works)

1. **Load/store patterns for SPIR-V** — `AMD::populateLoadStoreOpToLLVMPatterns` is skipped; relying on generic `triton::populateMemoryOpToLLVMPatterns`. Untested.
2. **ConvertLayoutOp for SPIR-V** — AMD version skipped; generic version registered. May break for MFMA/WMMA → blocked layout conversions.
3. **FP4 lowering** — `populateFp4ToFpToLLVMPatterns` is AMD-only, no SPIR-V equivalent wired yet.
4. **Barrier / TensorPtr / Masked patterns** — AMD versions skipped; generic coverage unverified.
5. **`make_llir()` AMD-specific attributes** (`amdgpu-waves-per-eu`, etc.) — harmless leakage but should be gated.
6. **SPV_NV_cooperative_matrix2** — not registered in LLVM backend. Blocks coop matrix reduce/advanced unary/binary.
7. **`muladd_scaled` (Set 3)** — no SPIR-V opcode exists at the spec level.

## Files changed this session (not committed)

### LLVM (`/home/mrsidims/LLVM/llvm-project`, branch `triton-compatible-spirv`)
All committed via cherry-pick except:
- `llvm/lib/Target/SPIRV/SPIRVEmitIntrinsics.cpp` — `TM->getSubtarget` fix (uncommitted, needs commit before switching branches)

### Triton (`/home/mrsidims/Triton/triton`, branch `main`, all uncommitted)
- `CMakeLists.txt`
- `python/triton/knobs.py`
- `third_party/amd/backend/compiler.py`
- `third_party/amd/lib/TritonAMDGPUToLLVM/CMakeLists.txt`
- `third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM.cpp`
- `third_party/amd/lib/TritonAMDGPUToLLVM/DotOpToLLVM/CoopMatrix.cpp` (new + TypeSwitch→isa fix)
- `third_party/amd/lib/TritonAMDGPUToLLVM/ElementwiseOpToLLVM.cpp`
- `third_party/amd/lib/TritonAMDGPUToLLVM/PatternTritonGPUOpToLLVM.h`
- `third_party/amd/lib/TritonAMDGPUToLLVM/SPIRVTargetInfo.cpp` (new)
- `third_party/amd/lib/TritonAMDGPUToLLVM/SPIRVTargetInfo.h` (new)
- `third_party/amd/lib/TritonAMDGPUToLLVM/TritonGPUToLLVM.cpp`
