//===-- RetargetCache.cpp - Caching layer for retargeted code objects ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RetargetCache.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"
#include <chrono>
#include <fstream>

using namespace llvm;
using namespace llvm::amdgpu;

RetargetCache &RetargetCache::getInstance() {
  static RetargetCache Instance;
  return Instance;
}

RetargetCache::RetargetCache() {
  // Check environment variables
  if (std::optional<std::string> Env =
          sys::Process::GetEnv("ROCM_RETARGET_DISABLE_CACHE")) {
    Enabled = (*Env != "1" && *Env != "true");
  }

  if (std::optional<std::string> Env =
          sys::Process::GetEnv("ROCM_RETARGET_VERBOSE")) {
    Verbose = (*Env == "1" || *Env == "true");
  }

  // Determine cache directory
  if (std::optional<std::string> Env =
          sys::Process::GetEnv("ROCM_RETARGET_CACHE")) {
    CacheDir = *Env;
  } else if (std::optional<std::string> Home = sys::Process::GetEnv("HOME")) {
    SmallString<256> DefaultPath(*Home);
    sys::path::append(DefaultPath, ".cache", "rocm-retarget");
    CacheDir = std::string(DefaultPath);
  } else {
    CacheDir = "/tmp/rocm-retarget";
  }

  if (Enabled) {
    initialize();
  }
}

RetargetCache::~RetargetCache() {
  if (Enabled && Initialized) {
    std::lock_guard<std::mutex> Lock(Mutex);
    saveIndex();
  }
}

void RetargetCache::initialize() {
  std::lock_guard<std::mutex> Lock(Mutex);
  if (Initialized)
    return;

  // Create cache directory structure
  SmallString<256> ObjectsDir(CacheDir);
  sys::path::append(ObjectsDir, "objects");

  std::error_code EC = sys::fs::create_directories(ObjectsDir);
  if (EC) {
    if (Verbose) {
      errs() << "rocm-retarget-cache: Failed to create cache directory: "
             << EC.message() << "\n";
    }
    Enabled = false;
    return;
  }

  // Load existing index
  loadIndex();

  Initialized = true;
  if (Verbose) {
    errs() << "rocm-retarget-cache: Initialized at " << CacheDir << "\n";
    errs() << "rocm-retarget-cache: " << CacheIndex.size()
           << " cached entries\n";
  }
}

std::string RetargetCache::computeHash(const void *Data, size_t Size) const {
  SHA256 Hasher;
  Hasher.update(ArrayRef<uint8_t>(static_cast<const uint8_t *>(Data), Size));
  auto Result = Hasher.final();

  // Convert to hex string
  std::string HashStr;
  raw_string_ostream OS(HashStr);
  for (uint8_t Byte : Result) {
    OS << format_hex_no_prefix(Byte, 2);
  }
  return HashStr;
}

std::string RetargetCache::makeCacheKey(const std::string &Hash,
                                        const std::string &TargetArch) const {
  return Hash + "_" + TargetArch;
}

std::string RetargetCache::getCachedFilePath(const std::string &Hash,
                                             const std::string &TargetArch) const {
  SmallString<256> Path(CacheDir);
  sys::path::append(Path, "objects", Hash.substr(0, 16) + "_" + TargetArch + ".co");
  return std::string(Path);
}

void RetargetCache::loadIndex() {
  SmallString<256> IndexPath(CacheDir);
  sys::path::append(IndexPath, "index.json");

  auto BufferOrErr = MemoryBuffer::getFile(IndexPath);
  if (!BufferOrErr) {
    // Index doesn't exist yet - that's OK
    return;
  }

  Expected<json::Value> JsonOrErr = json::parse((*BufferOrErr)->getBuffer());
  if (!JsonOrErr) {
    if (Verbose) {
      errs() << "rocm-retarget-cache: Failed to parse index: "
             << toString(JsonOrErr.takeError()) << "\n";
    }
    return;
  }

  json::Object *Root = JsonOrErr->getAsObject();
  if (!Root)
    return;

  json::Object *Entries = Root->getObject("entries");
  if (!Entries)
    return;

  for (auto &KV : *Entries) {
    if (json::Object *Entry = KV.second.getAsObject()) {
      if (auto Path = Entry->getString("path")) {
        CacheIndex[std::string(KV.first)] = std::string(*Path);
      }
    }
  }
}

void RetargetCache::saveIndex() {
  SmallString<256> IndexPath(CacheDir);
  sys::path::append(IndexPath, "index.json");

  std::error_code EC;
  raw_fd_ostream OS(IndexPath, EC, sys::fs::OF_Text);
  if (EC) {
    if (Verbose) {
      errs() << "rocm-retarget-cache: Failed to save index: " << EC.message()
             << "\n";
    }
    return;
  }

  json::Object Entries;
  for (const auto &KV : CacheIndex) {
    json::Object Entry;
    Entry["path"] = KV.second;
    Entries[KV.first] = std::move(Entry);
  }

  json::Object Root;
  Root["version"] = 1;
  Root["entries"] = std::move(Entries);

  OS << json::Value(std::move(Root)) << "\n";
}

std::optional<std::string> RetargetCache::lookup(const void *ElfData,
                                                 size_t ElfSize,
                                                 const std::string &TargetArch) {
  if (!Enabled)
    return std::nullopt;

  std::string Hash = computeHash(ElfData, ElfSize);
  std::string Key = makeCacheKey(Hash, TargetArch);

  std::lock_guard<std::mutex> Lock(Mutex);

  auto It = CacheIndex.find(Key);
  if (It != CacheIndex.end()) {
    SmallString<256> FullPath(CacheDir);
    sys::path::append(FullPath, It->second);

    // Verify the file exists
    if (sys::fs::exists(FullPath)) {
      ++Statistics.HitCount;
      if (Verbose) {
        errs() << "rocm-retarget-cache: Cache hit for " << TargetArch << "\n";
      }
      return std::string(FullPath);
    }
  }

  ++Statistics.MissCount;
  if (Verbose) {
    errs() << "rocm-retarget-cache: Cache miss for " << TargetArch << "\n";
  }
  return std::nullopt;
}

std::string RetargetCache::store(const void *OriginalElf, size_t OriginalSize,
                                 const void *RetargetedElf, size_t RetargetedSize,
                                 const std::string &SourceArch,
                                 const std::string &TargetArch) {
  if (!Enabled)
    return "";

  std::string Hash = computeHash(OriginalElf, OriginalSize);
  std::string Key = makeCacheKey(Hash, TargetArch);
  std::string FilePath = getCachedFilePath(Hash, TargetArch);

  // Write the retargeted code object
  std::error_code EC;
  raw_fd_ostream OS(FilePath, EC, sys::fs::OF_None);
  if (EC) {
    if (Verbose) {
      errs() << "rocm-retarget-cache: Failed to write cached file: "
             << EC.message() << "\n";
    }
    return "";
  }

  OS.write(static_cast<const char *>(RetargetedElf), RetargetedSize);
  OS.close();

  // Update the index
  {
    std::lock_guard<std::mutex> Lock(Mutex);

    // Store relative path
    SmallString<256> RelPath("objects");
    sys::path::append(RelPath, sys::path::filename(FilePath));
    CacheIndex[Key] = std::string(RelPath);

    ++Statistics.TotalEntries;
    Statistics.TotalSizeBytes += RetargetedSize;

    saveIndex();
  }

  if (Verbose) {
    errs() << "rocm-retarget-cache: Cached " << TargetArch << " at " << FilePath
           << "\n";
  }

  return FilePath;
}

std::vector<uint8_t> RetargetCache::read(const std::string &CachedPath) {
  auto BufferOrErr = MemoryBuffer::getFile(CachedPath);
  if (!BufferOrErr) {
    return {};
  }

  StringRef Content = (*BufferOrErr)->getBuffer();
  return std::vector<uint8_t>(Content.begin(), Content.end());
}

void RetargetCache::clear() {
  std::lock_guard<std::mutex> Lock(Mutex);

  SmallString<256> ObjectsDir(CacheDir);
  sys::path::append(ObjectsDir, "objects");

  std::error_code EC;
  for (sys::fs::directory_iterator I(ObjectsDir, EC), E; I != E && !EC;
       I.increment(EC)) {
    sys::fs::remove(I->path());
  }

  CacheIndex.clear();
  Statistics = Stats();
  saveIndex();

  if (Verbose) {
    errs() << "rocm-retarget-cache: Cache cleared\n";
  }
}

void RetargetCache::clearForTarget(const std::string &TargetArch) {
  std::lock_guard<std::mutex> Lock(Mutex);

  std::string Suffix = "_" + TargetArch;
  std::vector<std::string> KeysToRemove;

  for (const auto &KV : CacheIndex) {
    if (StringRef(KV.first).ends_with(Suffix)) {
      // Remove the file
      SmallString<256> FullPath(CacheDir);
      sys::path::append(FullPath, KV.second);
      sys::fs::remove(FullPath);
      KeysToRemove.push_back(KV.first);
    }
  }

  for (const auto &Key : KeysToRemove) {
    CacheIndex.erase(Key);
  }

  saveIndex();

  if (Verbose) {
    errs() << "rocm-retarget-cache: Cleared " << KeysToRemove.size()
           << " entries for " << TargetArch << "\n";
  }
}

RetargetCache::Stats RetargetCache::getStats() const {
  std::lock_guard<std::mutex> Lock(Mutex);
  return Statistics;
}

void RetargetCache::printStatus() const {
  std::lock_guard<std::mutex> Lock(Mutex);
  errs() << "ROCM Retarget Cache Status:\n";
  errs() << "  Directory: " << CacheDir << "\n";
  errs() << "  Enabled: " << (Enabled ? "yes" : "no") << "\n";
  errs() << "  Total entries: " << CacheIndex.size() << "\n";
  errs() << "  Cache hits: " << Statistics.HitCount << "\n";
  errs() << "  Cache misses: " << Statistics.MissCount << "\n";
}

// Runtime retargeting functions

std::vector<uint8_t> amdgpu::loadOrRetargetCodeObject(const void *ElfData,
                                                       size_t ElfSize,
                                                       const std::string &SourceArch,
                                                       const std::string &TargetArch) {
  // If same architecture, return as-is
  if (SourceArch == TargetArch) {
    return std::vector<uint8_t>(static_cast<const uint8_t *>(ElfData),
                                static_cast<const uint8_t *>(ElfData) + ElfSize);
  }

  RetargetCache &Cache = RetargetCache::getInstance();

  // Check cache first
  if (auto CachedPath = Cache.lookup(ElfData, ElfSize, TargetArch)) {
    return Cache.read(*CachedPath);
  }

  // Cache miss - run retargeter
  std::vector<uint8_t> Retargeted =
      runRetargeter(ElfData, ElfSize, SourceArch, TargetArch);

  if (Retargeted.empty()) {
    return {};
  }

  // Store in cache
  Cache.store(ElfData, ElfSize, Retargeted.data(), Retargeted.size(),
              SourceArch, TargetArch);

  return Retargeted;
}

std::vector<uint8_t> amdgpu::runRetargeter(const void *ElfData, size_t ElfSize,
                                           const std::string &SourceArch,
                                           const std::string &TargetArch) {
  // Create temporary files for input and output
  SmallString<128> InputPath, OutputPath;

  std::error_code EC;
  EC = sys::fs::createTemporaryFile("retarget_input", "co", InputPath);
  if (EC) {
    errs() << "rocm-retarget: Failed to create temp input file: "
           << EC.message() << "\n";
    return {};
  }

  EC = sys::fs::createTemporaryFile("retarget_output", "co", OutputPath);
  if (EC) {
    sys::fs::remove(InputPath);
    errs() << "rocm-retarget: Failed to create temp output file: "
           << EC.message() << "\n";
    return {};
  }

  // Write input
  {
    raw_fd_ostream OS(InputPath, EC, sys::fs::OF_None);
    if (EC) {
      sys::fs::remove(InputPath);
      sys::fs::remove(OutputPath);
      return {};
    }
    OS.write(static_cast<const char *>(ElfData), ElfSize);
  }

  // Find and run llvm-amdgpu-retarget
  ErrorOr<std::string> RetargetExe =
      sys::findProgramByName("llvm-amdgpu-retarget");
  if (!RetargetExe) {
    errs() << "rocm-retarget: llvm-amdgpu-retarget not found\n";
    sys::fs::remove(InputPath);
    sys::fs::remove(OutputPath);
    return {};
  }

  std::vector<StringRef> Args = {
      *RetargetExe,
      "--source=" + SourceArch,
      "--target=" + TargetArch,
      std::string(InputPath),
      "-o",
      std::string(OutputPath)};

  SmallString<256> ArgStr;
  for (const auto &Arg : Args) {
    ArgStr += Arg;
    ArgStr += " ";
  }

  int RetCode = sys::ExecuteAndWait(*RetargetExe, Args);
  sys::fs::remove(InputPath);

  if (RetCode != 0) {
    errs() << "rocm-retarget: Retargeting failed with code " << RetCode << "\n";
    sys::fs::remove(OutputPath);
    return {};
  }

  // Read output
  auto BufferOrErr = MemoryBuffer::getFile(OutputPath);
  sys::fs::remove(OutputPath);

  if (!BufferOrErr) {
    return {};
  }

  StringRef Content = (*BufferOrErr)->getBuffer();
  return std::vector<uint8_t>(Content.begin(), Content.end());
}
