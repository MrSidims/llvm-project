# RFC: Virtual File System (VFS) Support in LLD

## Problem Statement

### Background

LLVM has a mature Virtual File System abstraction (`llvm/Support/VirtualFileSystem.h`) that allows tools to virtualize all file I/O. Clang uses it extensively via `FileManager` to support features like VFS overlays, reproducible builds, and in-memory compilation. However, LLD — the LLVM linker — largely bypasses VFS and reads files directly from disk via `MemoryBuffer::getFile()`.

### Motivation: GPU Offloading Pipeline

The primary motivating use case is the GPU offloading pipeline in `clang-linker-wrapper`. Today, this pipeline:

1. Compiles device code to `.o` files, writing them to temporary files on disk
2. Spawns LLD as a subprocess to link these `.o` files
3. LLD reads them back from disk
4. The linked device binary is embedded into the host object

This creates unnecessary disk I/O overhead. With VFS support, the pipeline could instead:

1. Compile device code to in-memory buffers
2. Call `lldMain()` in-process, passing an `InMemoryFileSystem` containing the compiler outputs
3. Eliminate all intermediate temporary file creation and disk round-trips

### Current State

| Backend | VFS Status |
|---------|-----------|
| **COFF** | Partial — `/vfsoverlay` option exists but VFS is only used for file discovery (`vfs->status()`), not actual reads |
| **ELF** | None — uses `--chroot` and `--remap-inputs` as path-level alternatives |
| **Mach-O** | None |
| **WASM** | None |

The COFF backend's partial support is asymmetric: it uses VFS for `findFile()` path resolution but reads files directly from disk via `MemoryBuffer::getFile()`. This limits its usefulness to scenarios where file paths need redirection but the files themselves must exist on disk.

### Goals

1. **Library API**: Enable callers of `lldMain()` to pass a VFS containing input files, eliminating disk I/O for in-process linking
2. **Command-line**: Provide `--vfs-overlay` across all backends for YAML-based file redirection
3. **Backward compatibility**: Zero behavior change when VFS is not used (null VFS falls through to direct disk I/O)
4. **Consistency**: Unified VFS infrastructure shared across all four backends

### Non-Goals

- **Output VFS**: Output files (`-o`, import libraries, PDB files) continue to use direct disk I/O. Virtualizing outputs is a future phase with different requirements (streaming writes, `FileOutputBuffer`, `raw_fd_ostream`)
- **Replacing `--chroot`/`--remap-inputs`**: These ELF features operate at the path string level before VFS resolution. They remain useful and complementary

---

## Proposal

### API Changes

#### `lldMain()` Signature

```cpp
// Before
Result lldMain(ArrayRef<const char *> args, raw_ostream &stdoutOS,
               raw_ostream &stderrOS, ArrayRef<DriverDef> drivers);

// After
Result lldMain(ArrayRef<const char *> args, raw_ostream &stdoutOS,
               raw_ostream &stderrOS, ArrayRef<DriverDef> drivers,
               IntrusiveRefCntPtr<vfs::FileSystem> vfs = nullptr);
```

The default `nullptr` argument preserves source compatibility with all existing callers.

#### Backend `link()` Signatures

All backend `link()` functions gain a VFS parameter:

```cpp
// Via LLD_HAS_DRIVER macro
bool link(ArrayRef<const char *> args, raw_ostream &stdoutOS,
          raw_ostream &stderrOS, bool exitEarly, bool disableOutput,
          IntrusiveRefCntPtr<vfs::FileSystem> vfs = nullptr);
```

#### Usage Examples

```cpp
// Option 1: Default — reads from real disk, fully backward compatible
lldMain(args, stdout, stderr, drivers);

// Option 2: In-memory filesystem — all inputs from RAM
auto memFS = makeIntrusiveRefCnt<vfs::InMemoryFileSystem>();
memFS->addFile("/tmp/input.o", 0, MemoryBuffer::getMemBuffer(objData));
lldMain(args, stdout, stderr, drivers, memFS);

// Option 3: Overlay — in-memory inputs + real system libraries
auto overlay = makeIntrusiveRefCnt<vfs::OverlayFileSystem>(
    vfs::createPhysicalFileSystem());
overlay->pushOverlay(memFS);
lldMain(args, stdout, stderr, drivers, overlay);
```

### Command-Line Interface

All backends gain a `--vfs-overlay` option:

```
ld.lld --vfs-overlay overlay.yaml -o output input.o
ld64.lld --vfs-overlay overlay.yaml -o output input.o
wasm-ld --vfs-overlay overlay.yaml -o output input.o
lld-link /vfsoverlay:overlay.yaml  (existing, now uses shared infra)
```

When both a caller-provided VFS and `--vfs-overlay` are present, the YAML overlay is layered on top of the caller's VFS. This matches Clang's behavior.

---

## Design

### Architecture

```
                    lldMain(args, vfs)
                           |
                    DriverDispatcher
                           |
              +------------+------------+
              |            |            |
          elf::link   coff::link   macho::link  ...
              |            |            |
          ctx.vfs = vfs    |            |
              |            |            |
         --vfs-overlay?    |            |
              |            |            |
    createVFSFromOverlay   |            |
    (layers on ctx.vfs)    |            |
              |            |            |
         readFileVFS()  readFileVFS()  readFileVFS()
              |            |            |
         vfs != null?  vfs != null?  vfs != null?
           /    \        /    \        /    \
         VFS   disk    VFS   disk    VFS   disk
```

### VFS Storage

The VFS is stored in `CommonLinkerContext`, the base class for all backend contexts:

```cpp
class CommonLinkerContext {
public:
  // ...existing members...
  IntrusiveRefCntPtr<vfs::FileSystem> vfs;
};
```

- **ELF**: `Ctx` inherits `CommonLinkerContext`, so `ctx.vfs` is directly available
- **COFF**: `COFFLinkerContext` inherits `CommonLinkerContext`, same pattern
- **Mach-O**: Uses `CommonLinkerContext` directly (accessed via `commonContext().vfs`)
- **WASM**: Has its own `Ctx` struct (not inheriting `CommonLinkerContext`), so `vfs` is added as a direct member with explicit reset in `Ctx::reset()`

`IntrusiveRefCntPtr` is used (not `unique_ptr`) because the VFS may be shared across multiple LLD invocations from a library caller. This matches the pattern used throughout LLVM's VFS infrastructure and in Clang's `FileManager`.

### Null VFS Fast Path

When VFS is null (the default), helper functions short-circuit directly to the underlying system calls with zero overhead:

```cpp
ErrorOr<std::unique_ptr<MemoryBuffer>>
readFileVFS(vfs::FileSystem *vfs, const Twine &path,
            bool isText, bool requiresNullTerminator) {
  if (!vfs)
    return MemoryBuffer::getFile(path, isText, requiresNullTerminator);
  return vfs->getBufferForFile(path, ...);
}

bool existsVFS(vfs::FileSystem *vfs, const Twine &path) {
  if (!vfs)
    return sys::fs::exists(path);
  return vfs->exists(path);
}
```

This ensures existing users see no performance regression.

### Path Transform Ordering (ELF)

The ELF backend has existing path transformation features:

1. `--chroot`: Prepends a root directory to absolute paths
2. `--remap-inputs`: Pattern-based path rewriting

These operate on the path **string** before any file I/O. VFS then resolves the **transformed path**. This ordering is correct and requires no semantic changes:

```
original path  -->  --chroot  -->  --remap-inputs  -->  VFS resolution
"/lib/foo.o"       "/sysroot/lib/foo.o"                  vfs->getBufferForFile(...)
```

### Shared VFS Overlay Parsing

A common `createVFSFromOverlay()` function replaces the COFF-specific `getVFS()`:

```cpp
IntrusiveRefCntPtr<vfs::FileSystem>
createVFSFromOverlay(StringRef overlayPath,
                     IntrusiveRefCntPtr<vfs::FileSystem> baseFS,
                     function_ref<void(const Twine &)> errHandler);
```

- Reads the YAML overlay file through the base VFS (not direct disk I/O)
- Layers the overlay on top of the base filesystem
- If base is null, uses `createPhysicalFileSystem()` as default
- Error handling is delegated to backend-specific error reporters via callback

### Thread Safety (COFF)

The COFF backend reads files asynchronously via `std::async`. The VFS is captured by value (ref-counted) in the lambda:

```cpp
return std::async(strategy, [=, vfs = std::move(vfs)]() {
  return readFileVFS(vfs.get(), path, ...);
});
```

Using `createPhysicalFileSystem()` as the base (rather than `getRealFileSystem()`) ensures each thread has an independent CWD, avoiding races.

### What Is NOT Routed Through VFS

Certain file operations intentionally bypass VFS:

- **Output files**: `-o`, import libraries, PDB files use `FileOutputBuffer`/`raw_fd_ostream`
- **`--reproduce` tar**: Reads files for archival — uses real filesystem to capture actual file state
- **Temporary files**: Manifest generation, temp import libraries — locally generated artifacts
- **`rewritePath()`**: Checks file existence for `--reproduce` metadata

These are either output operations or operations where VFS indirection would be semantically incorrect.

---

## Files Changed

### Common Infrastructure
| File | Change |
|------|--------|
| `lld/include/lld/Common/CommonLinkerContext.h` | Added `vfs` member |
| `lld/include/lld/Common/Driver.h` | VFS parameter in `Driver`, `lldMain()`, `LLD_HAS_DRIVER` |
| `lld/Common/DriverDispatcher.cpp` | Forward VFS through dispatch |
| `lld/include/lld/Common/Filesystem.h` | `readFileVFS()`, `existsVFS()`, `createVFSFromOverlay()` |
| `lld/Common/Filesystem.cpp` | Implementation of VFS utilities |

### ELF Backend
| File | Change |
|------|--------|
| `lld/ELF/Driver.cpp` | Accept VFS, parse `--vfs-overlay` |
| `lld/ELF/Options.td` | `--vfs-overlay` option |
| `lld/ELF/InputFiles.cpp` | `readFile()` and dependent library through VFS |
| `lld/ELF/DriverUtils.cpp` | `findFile()`, `searchScript()` through VFS |
| `lld/ELF/LTO.cpp` | BB sections file through VFS |
| `lld/ELF/ScriptParser.cpp` | Linker script file discovery through VFS |

### COFF Backend
| File | Change |
|------|--------|
| `lld/COFF/Config.h` | Removed `vfs` field (moved to context) |
| `lld/COFF/Driver.cpp` | Replaced `getVFS()`, VFS in `createFutureForFile`, `findFile` |
| `lld/COFF/DriverUtils.cpp` | Section layout and DOS stub reads through VFS |
| `lld/COFF/SymbolTable.cpp` | Module def file through VFS |
| `lld/COFF/PDB.cpp` | Natvis and named stream files through VFS |

### Mach-O Backend
| File | Change |
|------|--------|
| `lld/MachO/Driver.cpp` | Accept VFS, parse `--vfs-overlay` |
| `lld/MachO/Options.td` | `--vfs-overlay` option |
| `lld/MachO/InputFiles.cpp` | `readFile()` through VFS |

### WASM Backend
| File | Change |
|------|--------|
| `lld/wasm/Config.h` | Added `vfs` field to `Ctx` |
| `lld/wasm/Driver.cpp` | Accept VFS, parse `--vfs-overlay`, reset in `Ctx::reset()` |
| `lld/wasm/Options.td` | `--vfs-overlay` option |
| `lld/wasm/InputFiles.cpp` | `readFile()` through VFS |

### MinGW
| File | Change |
|------|--------|
| `lld/MinGW/Driver.cpp` | Accept and forward VFS to `coff::link()` |

### Tests
| File | Change |
|------|--------|
| `lld/test/ELF/vfs-overlay.s` | New: basic overlay, library paths, error cases |
| `lld/test/COFF/vfsoverlay.test` | Updated error messages for shared infra |

---

## Future Work

1. **`clang-linker-wrapper` integration** (Phase 5): Call `lldMain()` in-process with `InMemoryFileSystem` containing device `.o` files, eliminating temporary disk I/O in the GPU offloading pipeline
2. **Output VFS**: Virtualize output file writes for fully in-memory linking pipelines
3. **Additional tests**: VFS interaction with LTO cache, thin archives, linker scripts with `INPUT()` directives
