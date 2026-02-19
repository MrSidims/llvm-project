//===- Filesystem.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_FILESYSTEM_H
#define LLD_FILESYSTEM_H

#include "lld/Common/LLVM.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>
#include <system_error>

namespace llvm::vfs {
class FileSystem;
} // namespace llvm::vfs

namespace lld {
void unlinkAsync(StringRef path);
std::error_code tryCreateFile(StringRef path);
std::unique_ptr<llvm::raw_fd_ostream> openFile(StringRef file);
std::unique_ptr<llvm::raw_fd_ostream> openLTOOutputFile(StringRef file);

// Read a file through VFS if available, otherwise fall back to
// MemoryBuffer::getFile(). When vfs is null, this is equivalent to a direct
// MemoryBuffer::getFile() call with zero overhead.
llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
readFileVFS(llvm::vfs::FileSystem *vfs, const llvm::Twine &path,
            bool isText = false, bool requiresNullTerminator = false);

// Check file existence through VFS if available, otherwise fall back to
// sys::fs::exists().
bool existsVFS(llvm::vfs::FileSystem *vfs, const llvm::Twine &path);

// Create a VFS from a YAML overlay file, layered on top of the given base
// filesystem. If baseFS is null, the real filesystem is used as base.
llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem>
createVFSFromOverlay(llvm::StringRef overlayPath,
                     llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> baseFS,
                     llvm::function_ref<void(const llvm::Twine &)> errHandler);
} // namespace lld

#endif
