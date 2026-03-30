//===-- RetargetCache.h - Caching layer for retargeted code objects ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the RetargetCache class which provides disk caching for
// retargeted AMDGPU code objects. This allows runtime retargeting of third-party
// binaries with caching to avoid redundant retargeting operations.
//
// Cache Structure:
//   $HOME/.cache/rocm-retarget/
//   +-- index.json             # Maps (input hash, target arch) -> cached file
//   +-- objects/
//       +-- <hash>_<arch>.co   # Retargeted code objects
//
// Environment Variables:
//   ROCM_RETARGET_CACHE        - Override cache directory location
//   ROCM_RETARGET_DISABLE_CACHE - Set to 1 to disable caching
//   ROCM_RETARGET_VERBOSE      - Set to 1 for verbose logging
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TOOLS_LLVM_AMDGPU_RETARGET_RUNTIME_RETARGETCACHE_H
#define LLVM_TOOLS_LLVM_AMDGPU_RETARGET_RUNTIME_RETARGETCACHE_H

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace llvm {
namespace amdgpu {

/// Cache for retargeted AMDGPU code objects.
///
/// This class provides thread-safe caching of retargeted code objects on disk.
/// It uses SHA-256 hashing to identify source code objects and maintains an
/// index file mapping (hash, target_arch) pairs to cached files.
class RetargetCache {
public:
  /// Get the singleton cache instance.
  static RetargetCache &getInstance();

  /// Check if caching is enabled.
  bool isEnabled() const { return Enabled; }

  /// Look up a retargeted code object in the cache.
  ///
  /// \param ElfData Pointer to the source ELF data.
  /// \param ElfSize Size of the source ELF data.
  /// \param TargetArch Target GPU architecture (e.g., "gfx942").
  /// \returns Path to the cached file if found, empty optional otherwise.
  std::optional<std::string> lookup(const void *ElfData, size_t ElfSize,
                                    const std::string &TargetArch);

  /// Store a retargeted code object in the cache.
  ///
  /// \param OriginalElf Pointer to the original ELF data.
  /// \param OriginalSize Size of the original ELF data.
  /// \param RetargetedElf Pointer to the retargeted ELF data.
  /// \param RetargetedSize Size of the retargeted ELF data.
  /// \param SourceArch Source GPU architecture.
  /// \param TargetArch Target GPU architecture.
  /// \returns Path to the cached file.
  std::string store(const void *OriginalElf, size_t OriginalSize,
                    const void *RetargetedElf, size_t RetargetedSize,
                    const std::string &SourceArch,
                    const std::string &TargetArch);

  /// Read a cached code object into memory.
  ///
  /// \param CachedPath Path returned by lookup().
  /// \returns Vector containing the code object data, or empty on failure.
  std::vector<uint8_t> read(const std::string &CachedPath);

  /// Clear all cached entries.
  void clear();

  /// Clear cached entries for a specific target architecture.
  void clearForTarget(const std::string &TargetArch);

  /// Get cache statistics.
  struct Stats {
    size_t TotalEntries = 0;
    size_t TotalSizeBytes = 0;
    size_t HitCount = 0;
    size_t MissCount = 0;
  };
  Stats getStats() const;

  /// Print cache status to stderr.
  void printStatus() const;

private:
  RetargetCache();
  ~RetargetCache();

  // Non-copyable
  RetargetCache(const RetargetCache &) = delete;
  RetargetCache &operator=(const RetargetCache &) = delete;

  /// Initialize the cache directory and load the index.
  void initialize();

  /// Compute SHA-256 hash of data.
  std::string computeHash(const void *Data, size_t Size) const;

  /// Generate cache key from hash and target architecture.
  std::string makeCacheKey(const std::string &Hash,
                           const std::string &TargetArch) const;

  /// Load the index file from disk.
  void loadIndex();

  /// Save the index file to disk.
  void saveIndex();

  /// Get the path to a cached file.
  std::string getCachedFilePath(const std::string &Hash,
                                const std::string &TargetArch) const;

  std::string CacheDir;
  bool Enabled = true;
  bool Verbose = false;
  bool Initialized = false;
  mutable std::mutex Mutex;

  /// Index mapping cache keys to relative file paths.
  std::unordered_map<std::string, std::string> CacheIndex;

  /// Statistics
  mutable Stats Statistics;
};

/// Load a code object, retargeting if necessary.
///
/// This is the main entry point for runtime retargeting. It:
/// 1. Checks if the code object needs retargeting
/// 2. Looks up in cache
/// 3. If not cached, runs llvm-amdgpu-retarget
/// 4. Caches the result
/// 5. Returns the retargeted data
///
/// \param ElfData Pointer to the original ELF data.
/// \param ElfSize Size of the original ELF data.
/// \param SourceArch Source GPU architecture (from ELF metadata).
/// \param TargetArch Target GPU architecture (current GPU).
/// \returns Vector containing the (possibly retargeted) code object data.
std::vector<uint8_t> loadOrRetargetCodeObject(const void *ElfData,
                                               size_t ElfSize,
                                               const std::string &SourceArch,
                                               const std::string &TargetArch);

/// Run the retargeter tool and return the result.
///
/// \param ElfData Pointer to the source ELF data.
/// \param ElfSize Size of the source ELF data.
/// \param SourceArch Source GPU architecture.
/// \param TargetArch Target GPU architecture.
/// \returns Vector containing the retargeted code object data.
std::vector<uint8_t> runRetargeter(const void *ElfData, size_t ElfSize,
                                    const std::string &SourceArch,
                                    const std::string &TargetArch);

} // namespace amdgpu
} // namespace llvm

#endif // LLVM_TOOLS_LLVM_AMDGPU_RETARGET_RUNTIME_RETARGETCACHE_H
