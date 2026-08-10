//===-- AMDGPUOptimizeBarriers.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Removes redundant fence instructions and workgroup barriers and narrows
/// fence sync scopes. Redundancy typically appears after inlining when
/// composed device functions each carry their own defensive synchronization.
///
/// A fence is removable when on one side of it every path reaches a covering
/// fence of at least equal strength before any access to memory that other
/// threads could observe. A workgroup barrier is removable in the same way
/// with respect to other workgroup barriers. In kernels the function entry
/// and exits act as covers for fences because under the HSA memory model the
/// dispatch packet performs a system scope acquire at launch and a system
/// scope release at completion. That is a runtime contract rather than an
/// LLVM IR guarantee. Boundaries never justify removal of an execution
/// barrier since kernel entry and exit do not rendezvous waves.
///
/// A fence with agent or wider scope whose reachable shared accesses on both
/// sides are all LDS is narrowed to workgroup scope since LDS is not visible
/// beyond the workgroup. Narrowing is skipped on subtargets with workgroup
/// clusters where LDS of one workgroup may be accessed from another.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/AMDGPUAddrSpace.h"
#include "llvm/Support/AtomicOrdering.h"
#include "llvm/Target/TargetMachine.h"

#define DEBUG_TYPE "amdgpu-optimize-barriers"

using namespace llvm;

namespace {

enum : unsigned { MemLDS = 1, MemGlobal = 2, MemAll = MemLDS | MemGlobal };

enum : unsigned {
  RankSingleThread = 0,
  RankWavefront = 1,
  RankWorkgroup = 2,
  RankCluster = 3,
  RankAgent = 4,
  RankSystem = 5
};

struct ScopeInfo {
  unsigned Rank;
  bool IsOneAs;
};

static std::optional<ScopeInfo> parseScope(SyncScope::ID SSID,
                                           LLVMContext &Ctx) {
  std::optional<StringRef> NameOpt = Ctx.getSyncScopeName(SSID);
  if (!NameOpt)
    return std::nullopt;
  StringRef Name = *NameOpt;
  ScopeInfo SI{0, false};
  if (Name == "one-as") {
    SI.IsOneAs = true;
    Name = "";
  } else if (Name.consume_back("-one-as")) {
    SI.IsOneAs = true;
  }
  std::optional<unsigned> Rank = StringSwitch<std::optional<unsigned>>(Name)
                                     .Case("", RankSystem)
                                     .Case("singlethread", RankSingleThread)
                                     .Case("wavefront", RankWavefront)
                                     .Case("workgroup", RankWorkgroup)
                                     .Case("cluster", RankCluster)
                                     .Case("agent", RankAgent)
                                     .Default(std::nullopt);
  if (!Rank)
    return std::nullopt;
  SI.Rank = *Rank;
  return SI;
}

static unsigned classifyAS(unsigned AS) {
  switch (AS) {
  case AMDGPUAS::LOCAL_ADDRESS:
    return MemLDS;
  case AMDGPUAS::GLOBAL_ADDRESS:
  case AMDGPUAS::REGION_ADDRESS:
  case AMDGPUAS::BUFFER_FAT_POINTER:
  case AMDGPUAS::BUFFER_RESOURCE:
  case AMDGPUAS::BUFFER_STRIDED_POINTER:
    return MemGlobal;
  case AMDGPUAS::CONSTANT_ADDRESS:
  case AMDGPUAS::CONSTANT_ADDRESS_32BIT:
  case AMDGPUAS::PRIVATE_ADDRESS:
    return 0;
  default:
    return MemAll;
  }
}

// Union of address space kinds an instruction may access that other threads
// could observe. Scheduling markers order nothing and report no access.
// Execution barriers report a full access when BarrierBlocks is set. A fence
// cover found beyond a barrier must not justify removal because the position
// of a fence relative to a barrier decides which accesses the barrier
// publishes. Narrowing walks may look through barriers since a barrier never
// makes LDS visible outside the workgroup.
static unsigned accessMask(const Instruction &I, bool BarrierBlocks) {
  if (isa<FenceInst>(&I))
    return 0;
  if (auto *CB = dyn_cast<CallBase>(&I)) {
    if (auto *II = dyn_cast<IntrinsicInst>(CB)) {
      switch (II->getIntrinsicID()) {
      case Intrinsic::amdgcn_s_barrier:
        return BarrierBlocks ? unsigned(MemAll) : 0u;
      case Intrinsic::amdgcn_s_barrier_signal:
      case Intrinsic::amdgcn_s_barrier_signal_var:
      case Intrinsic::amdgcn_s_barrier_signal_isfirst:
      case Intrinsic::amdgcn_s_barrier_init:
      case Intrinsic::amdgcn_s_barrier_join:
      case Intrinsic::amdgcn_s_barrier_wait:
      case Intrinsic::amdgcn_s_barrier_leave:
      case Intrinsic::amdgcn_s_wakeup_barrier:
      case Intrinsic::amdgcn_s_get_barrier_state:
      case Intrinsic::amdgcn_s_get_named_barrier_state:
      case Intrinsic::amdgcn_s_cluster_barrier:
        return MemAll;
      case Intrinsic::amdgcn_wave_barrier:
      case Intrinsic::amdgcn_sched_barrier:
      case Intrinsic::amdgcn_sched_group_barrier:
      case Intrinsic::amdgcn_iglp_opt:
        return 0;
      default:
        break;
      }
    }
    MemoryEffects ME = CB->getMemoryEffects();
    if (ME.doesNotAccessMemory())
      return 0;
    if (ME.onlyAccessesArgPointees()) {
      unsigned Mask = 0;
      for (const Use &U : CB->args())
        if (U->getType()->isPointerTy())
          Mask |= classifyAS(U->getType()->getPointerAddressSpace());
      return Mask;
    }
    return MemAll;
  }
  if (auto *LI = dyn_cast<LoadInst>(&I))
    return classifyAS(LI->getPointerAddressSpace());
  if (auto *SI = dyn_cast<StoreInst>(&I))
    return classifyAS(SI->getPointerAddressSpace());
  if (auto *RMW = dyn_cast<AtomicRMWInst>(&I))
    return classifyAS(RMW->getPointerAddressSpace());
  if (auto *CX = dyn_cast<AtomicCmpXchgInst>(&I))
    return classifyAS(CX->getPointerAddressSpace());
  if (I.mayReadOrWriteMemory() || I.mayHaveSideEffects())
    return MemAll;
  return 0;
}

static bool isWorkgroupBarrier(const Instruction &I) {
  auto *II = dyn_cast<IntrinsicInst>(&I);
  return II && II->getIntrinsicID() == Intrinsic::amdgcn_s_barrier;
}

class BarrierOptimizer {
public:
  BarrierOptimizer(Function &F, bool AllowNarrowing)
      : F(F), Ctx(F.getContext()), AllowNarrowing(AllowNarrowing),
        IsKernel(AMDGPU::isKernel(F.getCallingConv())) {}

  bool run();

private:
  Function &F;
  LLVMContext &Ctx;
  bool AllowNarrowing;
  bool IsKernel;

  using CoverFn = function_ref<bool(const Instruction &)>;

  DenseMap<unsigned, std::optional<ScopeInfo>> ScopeCache;

  std::optional<ScopeInfo> getScope(SyncScope::ID SSID) {
    auto [It, Inserted] = ScopeCache.try_emplace(SSID);
    if (Inserted)
      It->second = parseScope(SSID, Ctx);
    return It->second;
  }

  std::optional<unsigned> scanBackward(Instruction *From, CoverFn IsCover,
                                       bool BarrierBlocks, bool BoundaryCovers);
  std::optional<unsigned> scanForward(Instruction *From, CoverFn IsCover,
                                      bool BarrierBlocks, bool BoundaryCovers);
  bool tryRemove(Instruction *I, CoverFn IsCover, bool BoundaryCovers);
};

std::optional<unsigned> BarrierOptimizer::scanBackward(Instruction *From,
                                                       CoverFn IsCover,
                                                       bool BarrierBlocks,
                                                       bool BoundaryCovers) {
  unsigned Mask = 0;
  SmallPtrSet<BasicBlock *, 16> Visited;
  SmallVector<BasicBlock *, 16> Worklist;

  auto ScanRange = [&](BasicBlock::reverse_iterator Begin,
                       BasicBlock::reverse_iterator End) {
    for (auto It = Begin; It != End; ++It) {
      if (IsCover(*It))
        return true;
      Mask |= accessMask(*It, BarrierBlocks);
    }
    return false;
  };

  auto EnqueueOrFail = [&](BasicBlock *BB) {
    if (BB == &F.getEntryBlock())
      return BoundaryCovers && IsKernel;
    append_range(Worklist, predecessors(BB));
    return true;
  };

  BasicBlock *StartBB = From->getParent();
  if (!ScanRange(std::next(From->getReverseIterator()), StartBB->rend()))
    if (!EnqueueOrFail(StartBB))
      return std::nullopt;

  while (!Worklist.empty()) {
    if (Mask == MemAll)
      return Mask;
    BasicBlock *BB = Worklist.pop_back_val();
    if (!Visited.insert(BB).second)
      continue;
    if (ScanRange(BB->rbegin(), BB->rend()))
      continue;
    if (!EnqueueOrFail(BB))
      return std::nullopt;
  }
  return Mask;
}

std::optional<unsigned> BarrierOptimizer::scanForward(Instruction *From,
                                                      CoverFn IsCover,
                                                      bool BarrierBlocks,
                                                      bool BoundaryCovers) {
  unsigned Mask = 0;
  SmallPtrSet<BasicBlock *, 16> Visited;
  SmallVector<BasicBlock *, 16> Worklist;

  auto ScanRange = [&](BasicBlock::iterator Begin, BasicBlock::iterator End) {
    for (auto It = Begin; It != End; ++It) {
      if (IsCover(*It))
        return true;
      Mask |= accessMask(*It, BarrierBlocks);
    }
    return false;
  };

  auto EnqueueOrFail = [&](BasicBlock *BB) {
    Instruction *Term = BB->getTerminator();
    if (succ_empty(BB))
      return isa<UnreachableInst>(Term) || (BoundaryCovers && IsKernel);
    append_range(Worklist, successors(BB));
    return true;
  };

  BasicBlock *StartBB = From->getParent();
  if (!ScanRange(std::next(From->getIterator()), StartBB->end()))
    if (!EnqueueOrFail(StartBB))
      return std::nullopt;

  while (!Worklist.empty()) {
    if (Mask == MemAll)
      return Mask;
    BasicBlock *BB = Worklist.pop_back_val();
    if (!Visited.insert(BB).second)
      continue;
    if (ScanRange(BB->begin(), BB->end()))
      continue;
    if (!EnqueueOrFail(BB))
      return std::nullopt;
  }
  return Mask;
}

bool BarrierOptimizer::tryRemove(Instruction *I, CoverFn IsCover,
                                 bool BoundaryCovers) {
  std::optional<unsigned> Back =
      scanBackward(I, IsCover, /*BarrierBlocks=*/true, BoundaryCovers);
  if (Back && *Back == 0) {
    I->eraseFromParent();
    return true;
  }
  std::optional<unsigned> Fwd =
      scanForward(I, IsCover, /*BarrierBlocks=*/true, BoundaryCovers);
  if (Fwd && *Fwd == 0) {
    I->eraseFromParent();
    return true;
  }
  return false;
}

bool BarrierOptimizer::run() {
  SmallVector<CallInst *> Barriers;
  SmallVector<FenceInst *> Fences;
  for (Instruction &I : instructions(F)) {
    if (auto *FI = dyn_cast<FenceInst>(&I))
      Fences.push_back(FI);
    else if (isWorkgroupBarrier(I))
      Barriers.push_back(cast<CallInst>(&I));
  }
  if (Barriers.empty() && Fences.empty())
    return false;

  bool Changed = false;

  // Barriers are resolved before fences. A fence walk may then cross the
  // position of a removed barrier and find a cover or boundary beyond it.
  // That needs no cross phase invariant because every removal justifies
  // itself against the IR as it stands when its phase runs. Surviving
  // barriers still report MemAll to fence walks and defeat removal.
  for (CallInst *&B : Barriers) {
    auto IsCover = [B](const Instruction &I) {
      return &I != B && isWorkgroupBarrier(I);
    };
    if (tryRemove(B, IsCover, /*BoundaryCovers=*/false)) {
      B = nullptr;
      Changed = true;
    }
  }

  auto MakeFenceCover = [this](const FenceInst *Self,
                               const ScopeInfo &SelfScope) {
    return [this, Self, SelfScope](const Instruction &I) {
      if (&I == Self)
        return false;
      auto *FI = dyn_cast<FenceInst>(&I);
      if (!FI || FI->getMetadata(LLVMContext::MD_mmra))
        return false;
      std::optional<ScopeInfo> SI = getScope(FI->getSyncScopeID());
      if (!SI || SI->Rank < SelfScope.Rank ||
          (SI->IsOneAs && !SelfScope.IsOneAs))
        return false;
      return isAtLeastOrStrongerThan(FI->getOrdering(), Self->getOrdering());
    };
  };

  for (FenceInst *&FI : Fences) {
    std::optional<ScopeInfo> Scope = getScope(FI->getSyncScopeID());
    if (!Scope)
      continue;
    auto IsCover = MakeFenceCover(FI, *Scope);
    if (tryRemove(FI, IsCover, /*BoundaryCovers=*/true)) {
      FI = nullptr;
      Changed = true;
    }
  }

  if (!AllowNarrowing)
    return Changed;

  for (FenceInst *FI : Fences) {
    if (!FI)
      continue;
    std::optional<ScopeInfo> Scope = getScope(FI->getSyncScopeID());
    if (!Scope || Scope->Rank < RankAgent)
      continue;
    auto IsCover = MakeFenceCover(FI, *Scope);
    std::optional<unsigned> Back = scanBackward(FI, IsCover,
                                                /*BarrierBlocks=*/false,
                                                /*BoundaryCovers=*/true);
    if (!Back || (*Back & MemGlobal))
      continue;
    std::optional<unsigned> Fwd = scanForward(FI, IsCover,
                                              /*BarrierBlocks=*/false,
                                              /*BoundaryCovers=*/true);
    if (!Fwd || (*Fwd & MemGlobal))
      continue;
    StringRef NewScope = Scope->IsOneAs ? "workgroup-one-as" : "workgroup";
    FI->setSyncScopeID(Ctx.getOrInsertSyncScopeID(NewScope));
    Changed = true;
  }
  return Changed;
}

static bool runOptimizeBarriers(Function &F, const TargetMachine &TM) {
  if (!TM.getTargetTriple().isAMDGCN())
    return false;
  const GCNSubtarget &ST = TM.getSubtarget<GCNSubtarget>(F);
  bool AllowNarrowing = !ST.hasClusters();
  return BarrierOptimizer(F, AllowNarrowing).run();
}

class AMDGPUOptimizeBarriersLegacy : public FunctionPass {
public:
  static char ID;
  AMDGPUOptimizeBarriersLegacy() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
    if (skipFunction(F))
      return false;
    const TargetMachine &TM =
        getAnalysis<TargetPassConfig>().getTM<TargetMachine>();
    return runOptimizeBarriers(F, TM);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<TargetPassConfig>();
  }
};

} // namespace

char AMDGPUOptimizeBarriersLegacy::ID = 0;

char &llvm::AMDGPUOptimizeBarriersLegacyPassID =
    AMDGPUOptimizeBarriersLegacy::ID;

INITIALIZE_PASS_BEGIN(AMDGPUOptimizeBarriersLegacy, DEBUG_TYPE,
                      "AMDGPU Optimize Barriers", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(AMDGPUOptimizeBarriersLegacy, DEBUG_TYPE,
                    "AMDGPU Optimize Barriers", false, false)

FunctionPass *llvm::createAMDGPUOptimizeBarriersLegacyPass() {
  return new AMDGPUOptimizeBarriersLegacy();
}

PreservedAnalyses AMDGPUOptimizeBarriersPass::run(Function &F,
                                                  FunctionAnalysisManager &AM) {
  if (!runOptimizeBarriers(F, TM))
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
