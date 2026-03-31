# llvm-amdgpu-retarget Design Document

## Overview

`llvm-amdgpu-retarget` is a tool for retargeting AMDGPU code objects from one GPU
architecture to another (e.g., gfx942 → gfx90a). It provides two pipelines:

1. **MCInst Pipeline** (default) - Direct binary translation at MCInst level
2. **MIR Pipeline** - Lifts to MachineFunction for LLVM backend integration

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         llvm-amdgpu-retarget                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐                                                            │
│  │ Input ELF   │                                                            │
│  │ (gfx942)    │                                                            │
│  └──────┬──────┘                                                            │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │                    MC Disassembler                          │            │
│  │            (LLVM AMDGPU Disassembler)                       │            │
│  └──────────────────────────┬──────────────────────────────────┘            │
│                             │                                               │
│                             ▼                                               │
│                    ┌─────────────────┐                                      │
│                    │  MCInst Stream  │                                      │
│                    └────────┬────────┘                                      │
│                             │                                               │
│              ┌──────────────┴──────────────┐                                │
│              │                             │                                │
│              ▼                             ▼                                │
│  ┌───────────────────────┐    ┌───────────────────────┐                     │
│  │   MCInst Pipeline     │    │    MIR Pipeline       │                     │
│  │      (default)        │    │   (--use-mir)         │                     │
│  │                       │    │                       │                     │
│  │  AMDGPURetargeter     │    │  MIRLifter            │                     │
│  │  LivenessAnalyzer     │    │  BackendPipeline      │                     │
│  │  ELFRetargetWriter    │    │  RetargetPipeline     │                     │
│  └───────────┬───────────┘    └───────────┬───────────┘                     │
│              │                             │                                │
│              └──────────────┬──────────────┘                                │
│                             │                                               │
│                             ▼                                               │
│                    ┌─────────────────┐                                      │
│                    │   Output ELF    │                                      │
│                    │    (gfx90a)     │                                      │
│                    └─────────────────┘                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## MCInst Pipeline (Default)

Direct MCInst-to-MCInst translation with liveness-based register allocation.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MCInst Pipeline                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input: MCInst[] + Offsets[]                                                │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │              LivenessAnalyzer::analyze()                    │            │
│  │  ┌─────────────────────────────────────────────────────┐    │            │
│  │  │  1. buildCFG()         - Detect branches, build CFG │    │            │
│  │  │  2. getRegDefUse()     - Extract defs/uses per inst │    │            │
│  │  │  3. runDataflow()      - Backward dataflow analysis │    │            │
│  │  │  4. computeLiveness()  - Live-in/live-out sets      │    │            │
│  │  └─────────────────────────────────────────────────────┘    │            │
│  └──────────────────────────┬──────────────────────────────────┘            │
│                             │                                               │
│                             ▼                                               │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │             AMDGPURetargeter::transformAll()                │            │
│  │  ┌─────────────────────────────────────────────────────┐    │            │
│  │  │  For each instruction:                              │    │            │
│  │  │    1. Check if needs transformation                 │    │            │
│  │  │    2. Map opcode (source → target)                  │    │            │
│  │  │    3. If needs emulation:                           │    │            │
│  │  │       - allocateScratchVGPR() using liveness        │    │            │
│  │  │       - Emit emulation sequence                     │    │            │
│  │  │    4. Adjust branch offsets if size changed         │    │            │
│  │  └─────────────────────────────────────────────────────┘    │            │
│  └──────────────────────────┬──────────────────────────────────┘            │
│                             │                                               │
│                             ▼                                               │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │                   MCCodeEmitter                             │            │
│  │           Encode MCInst[] → raw bytes                       │            │
│  └──────────────────────────┬──────────────────────────────────┘            │
│                             │                                               │
│                             ▼                                               │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │                 ELFRetargetWriter                           │            │
│  │  ┌─────────────────────────────────────────────────────┐    │            │
│  │  │  1. Replace .text section with new bytes            │    │            │
│  │  │  2. Update e_flags (gfx942 → gfx90a)                │    │            │
│  │  │  3. Update .note.AMD metadata (MsgPack)             │    │            │
│  │  │  4. Adjust section offsets if size changed          │    │            │
│  │  │  5. Update kernel descriptor VGPRs if needed        │    │            │
│  │  └─────────────────────────────────────────────────────┘    │            │
│  └──────────────────────────┬──────────────────────────────────┘            │
│                             │                                               │
│                             ▼                                               │
│  Output: Retargeted ELF with updated .text, flags, metadata                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Liveness Analysis Detail

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Liveness Analysis                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input: Linear MCInst stream                                                │
│                                                                             │
│  Step 1: Build CFG                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                                                                     │    │
│  │    MCInst[0]  ──┐                                                   │    │
│  │    MCInst[1]    │  BB0                                              │    │
│  │    s_cbranch ───┼────────────────────┐                              │    │
│  │    MCInst[3]  ──┘         │          │                              │    │
│  │    MCInst[4]  ──┐         │          │                              │    │
│  │    MCInst[5]    │  BB1    │          │                              │    │
│  │    s_branch  ───┼─────────┼──┐       │                              │    │
│  │    MCInst[7]  ──┘         │  │       │                              │    │
│  │    MCInst[8]  ──┐         │  │       │  (branch target)             │    │
│  │    MCInst[9]    │  BB2 ◄──┘  │       │                              │    │
│  │    s_endpgm  ───┘            │       │                              │    │
│  │    MCInst[11] ──┐            │       │                              │    │
│  │    MCInst[12]   │  BB3 ◄─────┘───────┘                              │    │
│  │    s_endpgm  ───┘                                                   │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Step 2: Backward Dataflow (for each basic block)                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                                                                     │    │
│  │    LiveOut[BB] = ∪ LiveIn[Succ] for all successors                  │    │
│  │                                                                     │    │
│  │    For each instruction (bottom to top):                            │    │
│  │      LiveBefore[I] = (LiveAfter[I] - Defs[I]) ∪ Uses[I]             │    │
│  │                                                                     │    │
│  │    LiveIn[BB] = LiveBefore[first instruction]                       │    │
│  │                                                                     │    │
│  │    Iterate until fixed point                                        │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Step 3: Dead Register Query                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                                                                     │    │
│  │    DeadVGPRs[I] = AllVGPRs - LiveAfter[I]                           │    │
│  │                                                                     │    │
│  │    Example at instruction I:                                        │    │
│  │      LiveAfter[I] = {v0, v1, v2, v5}                                │    │
│  │      DeadVGPRs[I] = {v3, v4, v6, v7, ..., v255}                     │    │
│  │                                                                     │    │
│  │    → Can use v255, v254, etc. for scratch in emulation              │    │
│  │                                                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Instruction Emulation Example

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  v_lshl_add_u64 Emulation (gfx942 → gfx90a)                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Source (gfx942):                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  v_lshl_add_u64 v[4:5], v[0:1], v2, v[6:7]                          │    │
│  │                                                                     │    │
│  │  Semantics: v[4:5] = (v[0:1] << v2[4:0]) + v[6:7]                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                             │                                               │
│                             ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Scratch Allocation:                                                │    │
│  │    Query: getDeadVGPRsAt(instructionIndex)                          │    │
│  │    Result: {v8, v9, v10, ..., v255} are dead                        │    │
│  │    Allocate: scratch = v[254:255] (64-bit pair)                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                             │                                               │
│                             ▼                                               │
│  Target (gfx90a):                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  ; Emulation sequence (3 instructions)                              │    │
│  │  v_lshlrev_b64 v[254:255], v2, v[0:1]      ; tmp = src0 << src1     │    │
│  │  v_add_co_u32  v4, vcc, v254, v6           ; dst.lo = tmp.lo + src2 │    │
│  │  v_addc_co_u32 v5, vcc, v255, v7, vcc      ; dst.hi = tmp.hi + carry│    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## MIR Pipeline

Lifts MCInst to MachineFunction for potential LLVM backend integration.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            MIR Pipeline                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input: MCInst[] + Offsets[]                                                │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │                    MIRLifter                                │            │
│  │  ┌─────────────────────────────────────────────────────┐    │            │
│  │  │  Phase 1: CFG Construction                          │    │            │
│  │  │    - Scan for branches (s_branch, s_cbranch_*)      │    │            │
│  │  │    - Identify basic block leaders                   │    │            │
│  │  │    - Build predecessor/successor edges              │    │            │
│  │  └─────────────────────────────────────────────────────┘    │            │
│  │  ┌─────────────────────────────────────────────────────┐    │            │
│  │  │  Phase 2: MachineFunction Creation                  │    │            │
│  │  │    - Create LLVM IR Function stub                   │    │            │
│  │  │    - Create MachineModuleInfo                       │    │            │
│  │  │    - Create MachineBasicBlocks from CFG             │    │            │
│  │  └─────────────────────────────────────────────────────┘    │            │
│  │  ┌─────────────────────────────────────────────────────┐    │            │
│  │  │  Phase 3: Instruction Conversion                    │    │            │
│  │  │    - MCInst → MachineInstr via BuildMI              │    │            │
│  │  │    - Preserve physical registers                    │    │            │
│  │  │    - Convert operands (reg, imm, fp)                │    │            │
│  │  └─────────────────────────────────────────────────────┘    │            │
│  └──────────────────────────┬──────────────────────────────────┘            │
│                             │                                               │
│                             ▼                                               │
│                    ┌─────────────────┐                                      │
│                    │ MachineFunction │                                      │
│                    │  (phys regs)    │                                      │
│                    └────────┬────────┘                                      │
│                             │                                               │
│                             ▼                                               │
│  ┌─────────────────────────────────────────────────────────────┐            │
│  │                  BackendPipeline                            │            │
│  │  ┌─────────────────────────────────────────────────────┐    │            │
│  │  │  transformFunction()                                │    │            │
│  │  │    - Check each MachineInstr for compatibility      │    │            │
│  │  │    - Transform/expand incompatible instructions     │    │            │
│  │  └─────────────────────────────────────────────────────┘    │            │
│  │  ┌─────────────────────────────────────────────────────┐    │            │
│  │  │  emitCode()                                         │    │            │
│  │  │    - MachineInstr → MCInst conversion               │    │            │
│  │  │    - Emit via MCStreamer                            │    │            │
│  │  │    - Generate ELF with target flags                 │    │            │
│  │  └─────────────────────────────────────────────────────┘    │            │
│  └──────────────────────────┬──────────────────────────────────┘            │
│                             │                                               │
│                             ▼                                               │
│  Output: New ELF object file with target architecture                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### MIRLifter Detail: MCInst → MachineFunction

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MIRLifter Internals                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Input: MCInst Stream                                                │   │
│  │  ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐       │   │
│  │  │ I0 │ I1 │ I2 │ BR │ I4 │ I5 │ BR │ I7 │ I8 │ I9 │END │... │       │   │
│  │  └────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘       │   │
│  │   0x00 0x04 0x08 0x0C 0x10 0x14 0x18 0x1C 0x20 0x24 0x28              │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Step 1: Identify Leaders (basic block start addresses)             │   │
│  │                                                                      │   │
│  │  Leaders = { 0x00,           // First instruction                   │   │
│  │              0x10,           // After first branch                  │   │
│  │              0x1C,           // After second branch                 │   │
│  │              0x28 }          // Branch target                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Step 2: Create LiftedBasicBlocks                                   │   │
│  │                                                                      │   │
│  │  BB0: [I0, I1, I2, BR]  0x00-0x0C  → successors: {BB1, BB3}         │   │
│  │  BB1: [I4, I5, BR]      0x10-0x18  → successors: {BB2}              │   │
│  │  BB2: [I7, I8, I9, END] 0x1C-0x28  → successors: {}                 │   │
│  │  BB3: [...]             0x28-...   → successors: {...}              │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Step 3: Create MachineFunction Structure                           │   │
│  │                                                                      │   │
│  │  Module "lifted_module"                                              │   │
│  │    └── Function @retargeted_kernel (amdgpu_kernel)                  │   │
│  │          └── MachineFunction                                        │   │
│  │                ├── MBB0 (entry)                                     │   │
│  │                │     ├── successors: [MBB1, MBB3]                   │   │
│  │                │     └── instructions: [...]                        │   │
│  │                ├── MBB1                                             │   │
│  │                │     ├── predecessors: [MBB0]                       │   │
│  │                │     ├── successors: [MBB2]                         │   │
│  │                │     └── instructions: [...]                        │   │
│  │                ├── MBB2                                             │   │
│  │                │     ├── predecessors: [MBB1]                       │   │
│  │                │     └── instructions: [...]                        │   │
│  │                └── MBB3                                             │   │
│  │                      ├── predecessors: [MBB0]                       │   │
│  │                      └── instructions: [...]                        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Step 4: Convert Instructions (MCInst → MachineInstr)               │   │
│  │                                                                      │   │
│  │  MCInst:                         MachineInstr:                       │   │
│  │  ┌─────────────────────┐         ┌─────────────────────┐             │   │
│  │  │ Opcode: V_ADD_U32   │   →     │ Opcode: V_ADD_U32   │             │   │
│  │  │ Op[0]: Reg(VGPR0)   │         │ Op[0]: $vgpr0 (def) │             │   │
│  │  │ Op[1]: Reg(VGPR1)   │         │ Op[1]: $vgpr1       │             │   │
│  │  │ Op[2]: Imm(5)       │         │ Op[2]: 5            │             │   │
│  │  └─────────────────────┘         └─────────────────────┘             │   │
│  │                                                                      │   │
│  │  Key: Physical registers preserved (no virtual regs, no SSA)        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Code Emission (MachineInstr → MCInst → Bytes)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BackendPipeline::emitCode()                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  MachineFunction                                                            │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  For each MachineBasicBlock:                                        │    │
│  │    For each MachineInstr:                                           │    │
│  │      ┌─────────────────────────────────────────────────────────┐    │    │
│  │      │  MachineInstr                                           │    │    │
│  │      │  ┌─────────────────────────────────────────────────┐    │    │    │
│  │      │  │ Opcode: S_ADD_I32                               │    │    │    │
│  │      │  │ $sgpr2 = $sgpr2, $sgpr3                         │    │    │    │
│  │      │  └─────────────────────────────────────────────────┘    │    │    │
│  │      │                        │                                │    │    │
│  │      │                        ▼                                │    │    │
│  │      │  MCInst (built manually)                                │    │    │
│  │      │  ┌─────────────────────────────────────────────────┐    │    │    │
│  │      │  │ Opcode: S_ADD_I32                               │    │    │    │
│  │      │  │ Operands: [SGPR2, SGPR2, SGPR3]                 │    │    │    │
│  │      │  └─────────────────────────────────────────────────┘    │    │    │
│  │      │                        │                                │    │    │
│  │      │                        ▼                                │    │    │
│  │      │  MCStreamer::emitInstruction()                          │    │    │
│  │      │  ┌─────────────────────────────────────────────────┐    │    │    │
│  │      │  │ → MCCodeEmitter encodes to bytes                │    │    │    │
│  │      │  │ → MCAsmBackend handles fixups                   │    │    │    │
│  │      │  │ → MCObjectWriter writes to stream               │    │    │    │
│  │      │  └─────────────────────────────────────────────────┘    │    │    │
│  │      └─────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Output: ELF object file with .text section                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## ELF Rewriting

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ELFRetargetWriter                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input ELF (gfx942):                    Output ELF (gfx90a):                │
│  ┌─────────────────────┐                ┌─────────────────────┐             │
│  │ ELF Header          │                │ ELF Header          │             │
│  │   e_flags: 0x54C    │ ─────────────► │   e_flags: 0x53F    │  (updated)  │
│  ├─────────────────────┤                ├─────────────────────┤             │
│  │ Program Headers     │                │ Program Headers     │             │
│  │   (PT_LOAD, etc.)   │ ─────────────► │   (adjusted sizes)  │             │
│  ├─────────────────────┤                ├─────────────────────┤             │
│  │ .text section       │                │ .text section       │             │
│  │   (original code)   │ ─────────────► │   (retargeted)      │  (replaced) │
│  │   size: N bytes     │                │   size: M bytes     │             │
│  ├─────────────────────┤                ├─────────────────────┤             │
│  │ .note.AMD section   │                │ .note.AMD section   │             │
│  │   target: "gfx942"  │ ─────────────► │   target: "gfx90a"  │  (updated)  │
│  ├─────────────────────┤                ├─────────────────────┤             │
│  │ Other sections      │                │ Other sections      │             │
│  │   (.rodata, etc.)   │ ─────────────► │   (offset adjusted) │             │
│  ├─────────────────────┤                ├─────────────────────┤             │
│  │ Section Headers     │                │ Section Headers     │             │
│  │                     │ ─────────────► │   (sizes updated)   │             │
│  └─────────────────────┘                └─────────────────────┘             │
│                                                                             │
│  Key operations:                                                            │
│  1. Replace .text contents with retargeted code                             │
│  2. Update e_flags with target GPU (EF_AMDGPU_MACH_*)                       │
│  3. Parse & update .note.AMD MsgPack: amdhsa.target field                   │
│  4. If .text size changed (M ≠ N):                                          │
│     - Shift all following section offsets by (M - N)                        │
│     - Update program header sizes                                           │
│     - Update section header table offset (e_shoff)                          │
│  5. Update kernel descriptor VGPR count if emulation needs more             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
llvm/tools/llvm-amdgpu-retarget/
├── llvm-amdgpu-retarget.cpp    # Main tool, command-line handling
├── AMDGPURetargeter.h/cpp      # MCInst transformation, opcode mapping
├── LivenessAnalyzer.h/cpp      # CFG + dataflow liveness analysis
├── ELFRetargetWriter.h/cpp     # ELF parsing, rewriting, metadata update
├── MIRLifter.h/cpp             # MCInst → MachineFunction lifting
├── BackendPipeline.h/cpp       # MachineFunction transform + emission
├── RetargetPipeline.h/cpp      # Orchestrates MIR-based pipeline
├── CMakeLists.txt              # Build configuration
└── DESIGN.md                   # This file
```

---

## Usage

```bash
# Default MCInst pipeline (recommended)
llvm-amdgpu-retarget --source=gfx942 --target=gfx90a input.o -o output.o

# Verbose mode
llvm-amdgpu-retarget --source=gfx942 --target=gfx90a input.o -o output.o -v

# Dry run (analyze only)
llvm-amdgpu-retarget --source=gfx942 --target=gfx90a input.o --dry-run

# MIR pipeline
llvm-amdgpu-retarget --source=gfx942 --target=gfx90a input.o -o output.o --use-mir
```

---

## Future Work

1. **MIR Pipeline Enhancements**
   - Port instruction emulation from AMDGPURetargeter to BackendPipeline
   - Integrate with LLVM register allocator for spill handling
   - Use AsmPrinter for proper MCInstLowering

2. **Instruction Coverage**
   - Add emulation for more gfx950-only instructions
   - Support WMMA → MFMA transformation
   - Handle wave32 → wave64 conversion

3. **Verification**
   - Add numerical validation against reference execution
   - Integrate with LLVM test suite
