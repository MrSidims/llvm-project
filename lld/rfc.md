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
5. **Complete coverage**: All input file reads and path resolution (library search, framework lookup, script parsing, PDB discovery, `.imports` files) go through VFS

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

All backends support `--vfs-overlay`, and **multiple overlays** can be layered:

```
ld.lld --vfs-overlay a.yaml --vfs-overlay b.yaml -o output input.o
ld64.lld --vfs-overlay a.yaml --vfs-overlay b.yaml -o output input.o
wasm-ld --vfs-overlay a.yaml --vfs-overlay b.yaml -o output input.o
lld-link /vfsoverlay:a.yaml /vfsoverlay:b.yaml  (existing, now uses shared infra)
```

When both a caller-provided VFS and `--vfs-overlay` are present, each YAML overlay is layered on top of the previous VFS in command-line order. This matches Clang's behavior with multiple `-ivfsoverlay` flags.

---

## Design

### Architecture

```
                    lldMain(args, vfs)
                           |
                    DriverDispatcher
                           |
              +------+-----+------+------+
              |      |     |      |      |
          elf::link  |  coff::link | wasm::link
              |      |     |      |      |
         ctx.vfs = vfs     |   mingw::link
              |            |      |
       --vfs-overlay*?     |  ctx.vfs = vfs
              |            |      |
    createVFSFromOverlay   |   findFile()  -> existsVFS
    (layers each overlay)  |      |
              |         coff::link(vfs)
         readFileVFS()     |
         existsVFS()    readFileVFS()
              |         existsVFS()
         vfs != null?      |
           /    \     vfs != null?
         VFS   disk     /    \
                      VFS   disk
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
- **WASM**: Uses `CommonLinkerContext` directly (accessed via `commonContext().vfs`). The Wasm-specific `Ctx` struct does not duplicate the `vfs` field
- **MinGW**: Stores VFS in its `CommonLinkerContext` for search-path resolution, then forwards to `coff::link()`

`IntrusiveRefCntPtr` is used (not `unique_ptr`) because the VFS may be shared across multiple LLD invocations from a library caller. This matches the pattern used throughout LLVM's VFS infrastructure and in Clang's `FileManager`.

### Null VFS Fast Path

When VFS is null (the default), helper functions short-circuit directly to the underlying system calls with zero overhead:

```cpp
ErrorOr<std::unique_ptr<MemoryBuffer>>
readFileVFS(vfs::FileSystem *vfs, const Twine &path,
            bool isText, bool requiresNullTerminator, bool isVolatile) {
  if (!vfs)
    return MemoryBuffer::getFile(path, isText, requiresNullTerminator,
                                 isVolatile);
  return vfs->getBufferForFile(path, /*FileSize=*/-1,
                               requiresNullTerminator, isVolatile, isText);
}

bool existsVFS(vfs::FileSystem *vfs, const Twine &path) {
  if (!vfs)
    return sys::fs::exists(path);
  return vfs->exists(path);
}

bool isDirectoryVFS(vfs::FileSystem *vfs, const Twine &path) {
  if (!vfs)
    return sys::fs::is_directory(path);
  auto status = vfs->status(path);
  return status && status->isDirectory();
}
```

The `isVolatile` parameter is preserved from the original `MemoryBuffer::getFile()` calls. When `true`, it prevents mmap and forces a full read, which is important on Windows for files that may change during linking (order files, call graph files, module def files).

The `isDirectoryVFS` helper ensures that directory checks during search-path validation (e.g., Mach-O's `-L`/`-F` handling) also go through VFS, preventing virtual directories from being rejected as "non-directory".

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
- Returns `nullptr` on error (after calling `errHandler`), so callers must check before assigning. This prevents silently falling back to the base filesystem if a non-fatal error handler is used
- Error handling is delegated to backend-specific error reporters via callback
- Each backend iterates over all `--vfs-overlay` arguments, layering each on top of the previous

### Thread Safety (COFF)

The COFF backend reads files asynchronously via `std::async`. The VFS is captured by value (ref-counted) in the lambda:

```cpp
return std::async(strategy, [=, vfs = std::move(vfs)]() {
  return readFileVFS(vfs.get(), path, ...);
});
```

Using `createPhysicalFileSystem()` as the base (rather than `getRealFileSystem()`) ensures each thread has an independent CWD, avoiding races.

### Complete VFS Coverage by Backend

#### ELF
All input file paths go through VFS:
- `readFile()` — main file loading
- `findFile()`, `searchScript()`, `searchLibrary()` — library/script search paths
- `ScriptParser::addFile()` — linker script `INPUT()` / `GROUP()` directives
- LTO basic-block sections file

#### COFF
All input file paths go through VFS:
- `enqueuePath()` / `createFutureForFile()` — async file loading
- `findFile()` — library search with VFS status check
- `parseOrderFile()`, `parseCallGraphFile()` — with `isVolatile=true` preserved
- `parseModuleDefs()` — module definition files, `isVolatile=true` preserved
- `parseSectionLayout()`, `parseDosStub()` — section layout / DOS stub reads
- `PDB.cpp` — natvis and named stream files
- `findPdbPath()` — type-server PDB discovery (same folder, OBJ folder, output folder)
- `createManifestXmlWithInternalMt()` — manifest input files

#### Mach-O
All input file paths go through VFS:
- `readFile()` — main file loading
- `resolveDylibPath()` — `.tbd` / `.dylib` resolution
- `findPathCombination()` — library and framework search-path probing
- `findFramework()` — framework suffix resolution
- `warnIfNotDirectory()` — search-path existence and directory validation (via `existsVFS` + `isDirectoryVFS`)
- `getSearchPaths()` — `-L`/`-F` search-path and syslibroot directory probing (via `isDirectoryVFS`)
- `rewritePath()`, `rewriteInputPath()` — `--reproduce` path checks

#### WASM
All input file paths go through VFS:
- `readFile()` — main file loading
- `findFile()` — library search paths (`-L` / `-l`)
- `.imports` sidecar file discovery

#### MinGW
- `findFile()` / `searchLibrary()` — library search before forwarding to COFF
- VFS stored in `CommonLinkerContext` during MinGW's own processing phase, then forwarded to `coff::link()`

### What Is NOT Routed Through VFS

Certain file operations intentionally bypass VFS:

- **Output files**: `-o`, import libraries, PDB files use `FileOutputBuffer`/`raw_fd_ostream`
- **Temporary files**: Manifest generation temp files, temp import libraries — locally generated artifacts read back immediately
- **`TemporaryFile::getMemoryBuffer()`** (COFF): Reads linker-created temp files with `IsVolatile=true`
- **`createManifestXmlWithExternalMt()`** (COFF): Reads temp file output from `mt.exe`, not user input
- **`getModTime()`** (Mach-O): Queries real filesystem metadata (modification times) via `fs::status()`
- **`DependencyTracker` constructor** (Mach-O): Checks writability of an output path
- **`rewritePath()`** (ELF, COFF): Checks file existence for `--reproduce` tar metadata — captures real filesystem state for reproducibility
- **`fs::real_path()`** (Mach-O framework resolution): Symlink resolution requires real filesystem; the resolved path is then checked via VFS

These are either output operations, locally-generated temp files, or operations where VFS indirection would be semantically incorrect.

---

## Files Changed

### Common Infrastructure
| File | Change |
|------|--------|
| `lld/include/lld/Common/CommonLinkerContext.h` | Added `vfs` member |
| `lld/include/lld/Common/Driver.h` | VFS parameter in `Driver`, `lldMain()`, `LLD_HAS_DRIVER` |
| `lld/Common/DriverDispatcher.cpp` | Forward VFS through dispatch |
| `lld/include/lld/Common/Filesystem.h` | `readFileVFS()`, `existsVFS()`, `isDirectoryVFS()`, `createVFSFromOverlay()` |
| `lld/Common/Filesystem.cpp` | Implementation of VFS utilities |

### ELF Backend
| File | Change |
|------|--------|
| `lld/ELF/Driver.cpp` | Accept VFS, parse `--vfs-overlay` (multiple) |
| `lld/ELF/Options.td` | `--vfs-overlay` option |
| `lld/ELF/InputFiles.cpp` | `readFile()` and dependent library through VFS |
| `lld/ELF/DriverUtils.cpp` | `findFile()`, `searchScript()` through VFS |
| `lld/ELF/LTO.cpp` | BB sections file through VFS |
| `lld/ELF/ScriptParser.cpp` | Linker script file discovery through VFS |

### COFF Backend
| File | Change |
|------|--------|
| `lld/COFF/Config.h` | Removed `vfs` field (moved to context) |
| `lld/COFF/Driver.cpp` | Replaced `getVFS()`, VFS in `createFutureForFile`, `findFile`, overlay parsing with multiple support, `isVolatile=true` on order/callgraph files |
| `lld/COFF/DriverUtils.cpp` | Section layout, DOS stub, manifest input files through VFS |
| `lld/COFF/SymbolTable.cpp` | Module def file through VFS (`isVolatile=true`) |
| `lld/COFF/PDB.cpp` | Natvis and named stream files through VFS |
| `lld/COFF/InputFiles.cpp` | `findPdbPath()` through VFS |

### Mach-O Backend
| File | Change |
|------|--------|
| `lld/MachO/Driver.cpp` | Accept VFS, parse `--vfs-overlay` (multiple), `findFramework()` through VFS, `warnIfNotDirectory()` and `getSearchPaths()` through VFS (via `isDirectoryVFS`) |
| `lld/MachO/Options.td` | `--vfs-overlay` option |
| `lld/MachO/InputFiles.cpp` | `readFile()` through VFS |
| `lld/MachO/DriverUtils.cpp` | `resolveDylibPath()`, `findPathCombination()`, `rewritePath()`, `rewriteInputPath()` through VFS |

### WASM Backend
| File | Change |
|------|--------|
| `lld/wasm/Config.h` | Removed duplicate `vfs` field (uses `CommonLinkerContext::vfs`) |
| `lld/wasm/Driver.cpp` | Accept VFS, parse `--vfs-overlay` (multiple), `findFile()` and `.imports` discovery through VFS |
| `lld/wasm/Options.td` | `--vfs-overlay` option |
| `lld/wasm/InputFiles.cpp` | `readFile()` through VFS |

### MinGW
| File | Change |
|------|--------|
| `lld/MinGW/Driver.cpp` | Store VFS in context, `findFile()` through VFS, forward VFS to `coff::link()` |

### Tests
| File | Change |
|------|--------|
| `lld/test/ELF/vfs-overlay.s` | New: basic overlay, library paths, error cases |
| `lld/test/COFF/vfsoverlay.test` | Updated error messages for shared infra |
| `lld/test/MachO/vfs-overlay.s` | New: basic overlay, library search paths, error cases |
| `lld/test/wasm/vfs-overlay.s` | New: basic overlay, library search paths, error cases |

---

## Future Work

1. **`clang-linker-wrapper` integration**: Call `lldMain()` in-process with `InMemoryFileSystem` containing device `.o` files, eliminating temporary disk I/O in the GPU offloading pipeline
2. **Output VFS**: Virtualize output file writes for fully in-memory linking pipelines
3. **Additional tests**: VFS interaction with LTO cache, thin archives, linker scripts with `INPUT()` directives, MinGW library search through VFS
