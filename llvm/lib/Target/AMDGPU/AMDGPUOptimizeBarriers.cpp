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
/// threads could observe. The walks are direction aware. A pure acquire
/// fence can synchronize only through an atomic load sequenced before it and
/// a pure release fence only through an atomic store sequenced after it, so
/// on that side only atomic and volatile accesses and calls that may contain
/// them defeat removal. A covering fence may itself carry memory model
/// relaxation annotations when its known synchronize-as tags form a
/// superset of the tags on the candidate so it orders at least the address
/// spaces the candidate would. A workgroup barrier is removable in the same way
/// with respect to other workgroup barriers. In kernels the function entry
/// and exits act as covers for fences. A release fence that reaches kernel
/// exit without a later atomic store and an acquire fence that reaches
/// kernel entry without an earlier atomic load take part in no
/// synchronization and their removal needs no contract at all. The
/// remaining boundary cases rely on the HSA memory model where the dispatch
/// packet performs a system scope acquire at launch and a system scope
/// release at completion, so anything published before launch is already
/// visible to every wave and everything the kernel wrote is released at
/// completion. That is a runtime contract rather than an LLVM IR
/// guarantee. Kernel entry never justifies removal of an execution
/// barrier since launch does not rendezvous waves. Kernel exit justifies it
/// when no observable access follows on any path. A rendezvous no later
/// access can observe orders nothing, and removal takes the barrier from
/// every wave alike so no wave can be left waiting.
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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/MemoryModelRelaxationAnnotations.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/AMDGPUAddrSpace.h"
#include "llvm/Support/AtomicOrdering.h"
#include "llvm/Target/TargetMachine.h"

#define DEBUG_TYPE "amdgpu-optimize-barriers"

using namespace llvm;

namespace {

enum : unsigned { MemLDS = 1, MemGlobal = 2, MemAll = MemLDS | MemGlobal };

// Which accesses can take part in a synchronization the scanned fence
// provides. Volatile accesses always count because hand rolled flag
// protocols lean on fences around them.
enum class Relevance { All, AtomicReads, AtomicWrites };

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

// An untagged fence synchronizes every address space so it covers any
// candidate. A tagged fence is a valid cover only when every tag on both
// fences is a known synchronize-as tag and the candidate tags are a subset
// of the cover tags. The memory legalizer ignores unknown suffixes under
// its known prefix so a fence carrying any unknown tag may synchronize
// more than its known tags suggest and is never trusted as a cover.
static bool mmraCovers(const FenceInst *Cover, const FenceInst *Cand) {
  MMRAMetadata CoverTags(*Cover);
  if (CoverTags.empty())
    return true;
  MMRAMetadata CandTags(*Cand);
  if (CandTags.empty())
    return false;
  auto IsKnown = [](const MMRAMetadata::TagT &Tag) {
    return Tag.first == "amdgpu-synchronize-as" &&
           (Tag.second == "local" || Tag.second == "global");
  };
  if (!all_of(CoverTags, IsKnown) || !all_of(CandTags, IsKnown))
    return false;
  return all_of(CandTags, [&](const MMRAMetadata::TagT &Tag) {
    return CoverTags.hasTag(Tag.first, Tag.second);
  });
}

// Union of address space kinds an instruction may access that other threads
// could observe. Scheduling markers order nothing and report no access.
// Execution barriers report a full access when BarrierBlocks is set. A fence
// cover found beyond a barrier must not justify removal because the position
// of a fence relative to a barrier decides which accesses the barrier
// publishes. Narrowing walks may look through barriers since a barrier never
// makes LDS visible outside the workgroup. Under a refined relevance only
// accesses able to take part in the pairing the scanned fence provides are
// reported.
static unsigned accessMask(const Instruction &I, bool BarrierBlocks,
                           Relevance Rel) {
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
    if (Rel == Relevance::AtomicReads && !CB->mayReadFromMemory())
      return 0;
    if (Rel == Relevance::AtomicWrites && !CB->mayWriteToMemory())
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
  if (auto *LI = dyn_cast<LoadInst>(&I)) {
    if (Rel == Relevance::AtomicWrites && !LI->isVolatile())
      return 0;
    if (Rel == Relevance::AtomicReads && !LI->isAtomic() && !LI->isVolatile())
      return 0;
    return classifyAS(LI->getPointerAddressSpace());
  }
  if (auto *SI = dyn_cast<StoreInst>(&I)) {
    if (Rel == Relevance::AtomicReads && !SI->isVolatile())
      return 0;
    if (Rel == Relevance::AtomicWrites && !SI->isAtomic() && !SI->isVolatile())
      return 0;
    return classifyAS(SI->getPointerAddressSpace());
  }
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
                                       bool BarrierBlocks, bool BoundaryCovers,
                                       Relevance Rel);
  std::optional<unsigned> scanForward(Instruction *From, CoverFn IsCover,
                                      bool BarrierBlocks, bool BoundaryCovers,
                                      Relevance Rel);
  bool tryRemove(Instruction *I, CoverFn IsCover, bool BackBoundaryCovers,
                 bool FwdBoundaryCovers, Relevance BackRel, Relevance FwdRel);
};

std::optional<unsigned> BarrierOptimizer::scanBackward(Instruction *From,
                                                       CoverFn IsCover,
                                                       bool BarrierBlocks,
                                                       bool BoundaryCovers,
                                                       Relevance Rel) {
  unsigned Mask = 0;
  bool Justified = false;
  SmallPtrSet<BasicBlock *, 16> Visited;
  SmallVector<BasicBlock *, 16> Worklist;

  auto ScanRange = [&](BasicBlock::reverse_iterator Begin,
                       BasicBlock::reverse_iterator End) {
    for (auto It = Begin; It != End; ++It) {
      if (IsCover(*It)) {
        Justified = true;
        return true;
      }
      Mask |= accessMask(*It, BarrierBlocks, Rel);
    }
    return false;
  };

  auto EnqueueOrFail = [&](BasicBlock *BB) {
    if (BB == &F.getEntryBlock()) {
      if (!BoundaryCovers || !IsKernel)
        return false;
      Justified = true;
      return true;
    }
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
  // A drained worklist proves nothing by itself. Access free cycles visit
  // every block without ever reaching a cover or a boundary, so require
  // that at least one path was justified.
  if (!Justified)
    return std::nullopt;
  return Mask;
}

std::optional<unsigned> BarrierOptimizer::scanForward(Instruction *From,
                                                      CoverFn IsCover,
                                                      bool BarrierBlocks,
                                                      bool BoundaryCovers,
                                                      Relevance Rel) {
  unsigned Mask = 0;
  bool Justified = false;
  SmallPtrSet<BasicBlock *, 16> Visited;
  SmallVector<BasicBlock *, 16> Worklist;

  auto ScanRange = [&](BasicBlock::iterator Begin, BasicBlock::iterator End) {
    for (auto It = Begin; It != End; ++It) {
      if (IsCover(*It)) {
        Justified = true;
        return true;
      }
      Mask |= accessMask(*It, BarrierBlocks, Rel);
    }
    return false;
  };

  auto EnqueueOrFail = [&](BasicBlock *BB) {
    Instruction *Term = BB->getTerminator();
    if (succ_empty(BB)) {
      if (!isa<UnreachableInst>(Term) && (!BoundaryCovers || !IsKernel))
        return false;
      Justified = true;
      return true;
    }
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
  // A drained worklist proves nothing by itself. Access free cycles visit
  // every block without ever reaching a cover or a boundary, so require
  // that at least one path was justified.
  if (!Justified)
    return std::nullopt;
  return Mask;
}

bool BarrierOptimizer::tryRemove(Instruction *I, CoverFn IsCover,
                                 bool BackBoundaryCovers,
                                 bool FwdBoundaryCovers, Relevance BackRel,
                                 Relevance FwdRel) {
  std::optional<unsigned> Back = scanBackward(
      I, IsCover, /*BarrierBlocks=*/true, BackBoundaryCovers, BackRel);
  if (Back && *Back == 0) {
    I->eraseFromParent();
    return true;
  }
  std::optional<unsigned> Fwd = scanForward(
      I, IsCover, /*BarrierBlocks=*/true, FwdBoundaryCovers, FwdRel);
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
    if (tryRemove(B, IsCover, /*BackBoundaryCovers=*/false,
                  /*FwdBoundaryCovers=*/true, Relevance::All,
                  Relevance::All)) {
      B = nullptr;
      Changed = true;
    }
  }

  // Narrowing walks stop at a cover on the promise that accesses beyond it
  // are ordered by the cover at the candidate scope in every address space
  // so only untagged covers qualify there.
  auto MakeFenceCover = [this](const FenceInst *Self,
                               const ScopeInfo &SelfScope,
                               bool AllowTaggedCovers) {
    return [this, Self, SelfScope, AllowTaggedCovers](const Instruction &I) {
      if (&I == Self)
        return false;
      auto *FI = dyn_cast<FenceInst>(&I);
      if (!FI)
        return false;
      if (AllowTaggedCovers ? !mmraCovers(FI, Self)
                            : FI->hasMetadata(LLVMContext::MD_mmra))
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
    auto IsCover = MakeFenceCover(FI, *Scope, /*AllowTaggedCovers=*/true);
    AtomicOrdering Ord = FI->getOrdering();
    Relevance BackRel =
        isReleaseOrStronger(Ord) ? Relevance::All : Relevance::AtomicReads;
    Relevance FwdRel =
        isAcquireOrStronger(Ord) ? Relevance::All : Relevance::AtomicWrites;
    if (tryRemove(FI, IsCover, /*BackBoundaryCovers=*/true,
                  /*FwdBoundaryCovers=*/true, BackRel, FwdRel)) {
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
    auto IsCover = MakeFenceCover(FI, *Scope, /*AllowTaggedCovers=*/false);
    std::optional<unsigned> Back =
        scanBackward(FI, IsCover, /*BarrierBlocks=*/false,
                     /*BoundaryCovers=*/true, Relevance::All);
    if (!Back || (*Back & MemGlobal))
      continue;
    std::optional<unsigned> Fwd =
        scanForward(FI, IsCover, /*BarrierBlocks=*/false,
                    /*BoundaryCovers=*/true, Relevance::All);
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
