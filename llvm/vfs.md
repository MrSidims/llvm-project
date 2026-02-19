# LLVM Virtual File System (VFS) — Explained Simply

## What is a Virtual File System (VFS)?

Think of VFS as a **fake file system that sits between your program and the real disk**. When your program says "open file `/foo/bar.h`", instead of always going to the actual disk, the VFS intercepts the call and can:

1. **Return a file from memory** (never touches disk at all)
2. **Redirect to a different file on disk** (you ask for `/a/b.h`, it gives you `/x/y.h`)
3. **Fall through to the real disk** if it doesn't know the file

It's like a receptionist that intercepts your mail — sometimes she hands you a memo from her desk (in-memory), sometimes she reroutes you to a different mailbox (redirect), and sometimes she just says "go check the real mailbox" (pass-through).

---

## The LLVM VFS Class Hierarchy

Everything lives in `llvm/include/llvm/Support/VirtualFileSystem.h` and the implementation in `llvm/lib/Support/VirtualFileSystem.cpp`. The key classes:

```
FileSystem (abstract base)
├── RealFileSystem        — Talks to the actual OS disk
├── InMemoryFileSystem    — Files exist only in RAM
├── OverlayFileSystem     — Stack of FSes, first match wins
├── ProxyFileSystem       — Decorator: wraps another FS, overrides selectively
└── RedirectingFileSystem — YAML-configured path remapping
```

### What each does

**`RealFileSystem`** — The boring one. It just calls the OS. `open()`, `stat()`, etc. This is what you get if nobody sets up a VFS.

**`InMemoryFileSystem`** — The cool one. You programmatically add files:
```cpp
auto MemFS = makeIntrusiveRefCnt<vfs::InMemoryFileSystem>();
MemFS->addFile("/fake/path/foo.o", timestamp, memoryBuffer);
```
Now when someone does `openFileForRead("/fake/path/foo.o")`, they get that buffer back — no disk involved. Internally it builds a tree of `InMemoryDirectory` → `InMemoryFile` nodes, each `InMemoryFile` holding a `MemoryBuffer`.

**`OverlayFileSystem`** — A stack. You push file systems on top of each other:
```
┌─────────────────────┐
│  InMemoryFileSystem  │  ← checked first
├─────────────────────┤
│  RealFileSystem      │  ← fallback
└─────────────────────┘
```
When you look up a file, it checks top-to-bottom. First hit wins. This is the key composition mechanism — you overlay fake files on top of the real disk.

**`RedirectingFileSystem`** — Configured via a YAML file. Maps virtual paths to real paths:
```yaml
{ 'version': 0,
  'roots': [
    { 'type': 'file',
      'name': '/virtual/input.o',
      'external-contents': '/real/path/input.o' }
  ]}
```
Has three modes: `fallthrough` (try virtual, then real), `fallback` (try real, then virtual), `redirect-only` (only virtual).

**`ProxyFileSystem`** — Just forwards everything to a wrapped FS. You subclass it and override only the methods you care about.

---

## How Clang Uses It

Clang's `FileManager` (in `clang/include/clang/Basic/FileManager.h`) holds an `IntrusiveRefCntPtr<llvm::vfs::FileSystem>`. Every file access in Clang — opening headers, reading source files, PCH files — goes through this VFS.

The typical setup:
```cpp
// Default: just the real disk
auto FS = vfs::getRealFileSystem();

// Or: overlay an InMemoryFS on top of the real one
auto Overlay = makeIntrusiveRefCnt<vfs::OverlayFileSystem>(vfs::getRealFileSystem());
auto MemFS = makeIntrusiveRefCnt<vfs::InMemoryFileSystem>();
MemFS->addFile(...);
Overlay->pushOverlay(MemFS);

// Or: load a YAML VFS overlay
auto RedirectFS = vfs::getVFSFromYAML(yamlBuffer, handler, yamlPath);
```

This is how Clang can:
- Do **reproducible builds** (remap dependency paths)
- Run **tests without writing temp files** to disk
- Support **module caching** where files are virtualized

---

## Current State of lld

lld has **minimal** VFS support — only in the COFF linker (`lld/COFF/`):
- It stores a `std::unique_ptr<llvm::vfs::FileSystem> vfs` in its config
- It supports `--vfs-overlay` to load a YAML overlay
- But it's superficial — most file I/O in lld still uses raw `llvm::MemoryBuffer::getFile()` or `sys::fs::` calls directly, bypassing any VFS

The ELF and Mach-O linkers have essentially **zero** VFS integration.

---

## What Would "Adding VFS to lld" Mean?

### Step 1: Route all file I/O through VFS

Right now lld does things like:
```cpp
// Direct disk access — bypasses VFS
auto bufOrErr = MemoryBuffer::getFile(path);
```

This needs to become:
```cpp
// Goes through VFS — could be in-memory, redirected, or real
auto fileOrErr = FS->openFileForRead(path);
auto bufOrErr = (*fileOrErr)->getBuffer(path);
```

Every place lld opens a file (input `.o` files, linker scripts, archive libraries, version scripts, etc.) needs to go through the VFS instead of directly to disk.

### Step 2: Thread the VFS through lld's infrastructure

You need to pass the `FileSystem` reference to every component that reads files:
- The driver (command-line parsing, response files)
- The input file reader
- The linker script parser
- Archive handling
- Symbol resolution that opens lazy archives

### Step 3: Enable InMemoryFileSystem for the Code Object Manager use case

This is the payoff. Instead of:
```
compiler → writes .o to disk → linker reads .o from disk
```

You get:
```
compiler → puts .o in InMemoryFileSystem → linker reads .o from same InMemoryFS
```

No disk I/O for intermediate files. The compiler and linker share a VFS where object files exist only in RAM.

---

## The Nuances

### Why not just use pipes/stdin?
Pipes are sequential — you can only read once, and you need to know the order. With a VFS, files have names and can be accessed randomly, just like disk. The linker can open `foo.o`, then `bar.o`, then go back to `foo.o` — impossible with a pipe.

### Why is this in LLVM core and not Clang?
It **already is** in LLVM core (`llvm/Support/VirtualFileSystem.h`). Clang just uses it. The challenge is that lld mostly ignores it and does raw disk I/O.

### Reference counting matters
`FileSystem` uses `IntrusiveRefCntPtr` (like a shared_ptr). This means multiple components can share the same VFS without lifetime issues. When the compiler and linker share an `InMemoryFileSystem`, reference counting ensures it stays alive as long as either needs it.

### The OverlayFS pattern is key
You rarely use `InMemoryFileSystem` alone. The typical pattern:

```
OverlayFileSystem
├── InMemoryFileSystem  (compiler outputs live here)
└── RealFileSystem      (system libraries, linker scripts still on disk)
```

This way the linker can find compiler-generated `.o` files in memory but still access `/usr/lib/libc.a` from disk.

### UniqueID is how deduplication works
Each in-memory file gets a unique ID computed from `hash(parent_id, name, contents)`. This is how the system knows two paths point to the same file (like hardlinks). If you add the same content under two names, it still works correctly.

---

## TL;DR

| Concept | Analogy |
|---|---|
| `RealFileSystem` | Going to the filing cabinet yourself |
| `InMemoryFileSystem` | Someone hands you the document from memory |
| `OverlayFileSystem` | Check the desk first, then the filing cabinet |
| `RedirectingFileSystem` | A forwarding address ("file X is actually at path Y") |
| Adding VFS to lld | Replace every `open(path)` in lld with `vfs->open(path)` |
| The Code Object Manager goal | Compiler and linker share a virtual filing cabinet in RAM — no paper (disk) needed |
