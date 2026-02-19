commit 309dfb4e2a19fd047f3287451da99a52fa43a0ad
Author: Dmitry Sidorov <Dmitry.Sidorov@amd.com>
Date:   Thu Feb 19 12:05:28 2026 +0100

    wip

diff --git a/lld/COFF/Config.h b/lld/COFF/Config.h
index 1c0f874ddfd7..60a07f1b4740 100644
--- a/lld/COFF/Config.h
+++ b/lld/COFF/Config.h
@@ -293,9 +293,6 @@ struct Configuration {
   // Used for /print-symbol-order:
   StringRef printSymbolOrder;
 
-  // Used for /vfsoverlay:
-  std::unique_ptr<llvm::vfs::FileSystem> vfs;
-
   uint64_t align = 4096;
   uint64_t imageBase = -1;
   uint64_t fileAlign = 512;
diff --git a/lld/COFF/Driver.cpp b/lld/COFF/Driver.cpp
index 699d53ca6d2e..923e4e1b81fd 100644
--- a/lld/COFF/Driver.cpp
+++ b/lld/COFF/Driver.cpp
@@ -84,10 +84,13 @@ uint64_t coff::errCount(COFFLinkerContext &ctx) { return ctx.e.errorCount; }
 namespace lld::coff {
 
 bool link(ArrayRef<const char *> args, llvm::raw_ostream &stdoutOS,
-          llvm::raw_ostream &stderrOS, bool exitEarly, bool disableOutput) {
+          llvm::raw_ostream &stderrOS, bool exitEarly, bool disableOutput,
+          llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> vfs) {
   // This driver-specific context will be freed later by unsafeLldMain().
   auto *ctx = new COFFLinkerContext;
 
+  ctx->vfs = std::move(vfs);
+
   ctx->e.initialize(stdoutOS, stderrOS, exitEarly, disableOutput);
   ctx->e.logName = args::getFilenameWithoutExe(args[0]);
   ctx->e.errorLimitExceededMsg = "too many errors emitted, stopping now"
@@ -152,8 +155,9 @@ using MBErrPair = std::pair<std::unique_ptr<MemoryBuffer>, std::error_code>;
 
 // Create a std::future that opens and maps a file using the best strategy for
 // the host platform.
-static std::future<MBErrPair> createFutureForFile(std::string path,
-                                                  bool prefetchInputs) {
+static std::future<MBErrPair>
+createFutureForFile(std::string path, bool prefetchInputs,
+                    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> vfs) {
 #if _WIN64
   // On Windows, file I/O is relatively slow so it is best to do this
   // asynchronously.  But 32-bit has issues with potentially launching tons
@@ -162,9 +166,9 @@ static std::future<MBErrPair> createFutureForFile(std::string path,
 #else
   auto strategy = std::launch::deferred;
 #endif
-  return std::async(strategy, [=]() {
-    auto mbOrErr = MemoryBuffer::getFile(path, /*IsText=*/false,
-                                         /*RequiresNullTerminator=*/false);
+  return std::async(strategy, [=, vfs = std::move(vfs)]() {
+    auto mbOrErr = lld::readFileVFS(vfs.get(), path, /*isText=*/false,
+                                    /*requiresNullTerminator=*/false);
     if (!mbOrErr)
       return MBErrPair{nullptr, mbOrErr.getError()};
     // Prefetch memory pages in the background as we will need them soon enough.
@@ -365,8 +369,8 @@ void LinkerDriver::handleReproFile(StringRef path, InputOpt inputOpt) {
 }
 
 void LinkerDriver::enqueuePath(StringRef path, bool lazy, InputOpt inputOpt) {
-  auto future = std::make_shared<std::future<MBErrPair>>(
-      createFutureForFile(std::string(path), ctx.config.prefetchInputs));
+  auto future = std::make_shared<std::future<MBErrPair>>(createFutureForFile(
+      std::string(path), ctx.config.prefetchInputs, ctx.vfs));
   std::string pathStr = std::string(path);
   enqueueTask([=]() {
     llvm::TimeTraceScope timeScope("File: ", path);
@@ -380,8 +384,9 @@ void LinkerDriver::enqueuePath(StringRef path, bool lazy, InputOpt inputOpt) {
       // before something we can find with an architecture, we won't find the
       // winsysroot file.
       if (std::optional<StringRef> retryPath = findFileIfNew(pathStr)) {
-        auto retryMb = MemoryBuffer::getFile(*retryPath, /*IsText=*/false,
-                                             /*RequiresNullTerminator=*/false);
+        auto retryMb = lld::readFileVFS(ctx.vfs.get(), *retryPath,
+                                        /*isText=*/false,
+                                        /*requiresNullTerminator=*/false);
         ec = retryMb.getError();
         if (!ec) {
           mb = std::move(*retryMb);
@@ -487,7 +492,7 @@ void LinkerDriver::enqueueArchiveMember(const Archive::Child &c,
             "could not get the filename for the member defining symbol " +
                 toCOFFString(ctx, sym));
   auto future = std::make_shared<std::future<MBErrPair>>(
-      createFutureForFile(childName, ctx.config.prefetchInputs));
+      createFutureForFile(childName, ctx.config.prefetchInputs, ctx.vfs));
   enqueueTask([=]() {
     auto mbOrErr = future->get();
     if (mbOrErr.second)
@@ -632,8 +637,8 @@ void LinkerDriver::parseDirectives(InputFile *file) {
 // care of that. Note that the returned path is not guaranteed to exist.
 StringRef LinkerDriver::findFile(StringRef filename) {
   auto getFilename = [this](StringRef filename) -> StringRef {
-    if (ctx.config.vfs)
-      if (auto statOrErr = ctx.config.vfs->status(filename))
+    if (ctx.vfs)
+      if (auto statOrErr = ctx.vfs->status(filename))
         return saver().save(statOrErr->getName());
     return filename;
   };
@@ -645,12 +650,12 @@ StringRef LinkerDriver::findFile(StringRef filename) {
     SmallString<128> path = dir;
     sys::path::append(path, filename);
     path = SmallString<128>{getFilename(path.str())};
-    if (sys::fs::exists(path.str()))
+    if (lld::existsVFS(ctx.vfs.get(), path.str()))
       return saver().save(path.str());
     if (!hasExt) {
       path.append(".obj");
       path = SmallString<128>{getFilename(path.str())};
-      if (sys::fs::exists(path.str()))
+      if (lld::existsVFS(ctx.vfs.get(), path.str()))
         return saver().save(path.str());
     }
   }
@@ -1140,9 +1145,8 @@ void LinkerDriver::parseOrderFile(StringRef arg) {
   // Open a file.
   StringRef path = arg.substr(1);
   std::unique_ptr<MemoryBuffer> mb =
-      CHECK(MemoryBuffer::getFile(path, /*IsText=*/false,
-                                  /*RequiresNullTerminator=*/false,
-                                  /*IsVolatile=*/true),
+      CHECK(lld::readFileVFS(ctx.vfs.get(), path, /*isText=*/false,
+                             /*requiresNullTerminator=*/false),
             "could not open " + path);
 
   // Parse a file. An order file contains one symbol per line.
@@ -1168,9 +1172,8 @@ void LinkerDriver::parseOrderFile(StringRef arg) {
 
 void LinkerDriver::parseCallGraphFile(StringRef path) {
   std::unique_ptr<MemoryBuffer> mb =
-      CHECK(MemoryBuffer::getFile(path, /*IsText=*/false,
-                                  /*RequiresNullTerminator=*/false,
-                                  /*IsVolatile=*/true),
+      CHECK(lld::readFileVFS(ctx.vfs.get(), path, /*isText=*/false,
+                             /*requiresNullTerminator=*/false),
             "could not open " + path);
 
   // Build a map from symbol name to section.
@@ -1521,26 +1524,16 @@ std::optional<std::string> getReproduceFile(const opt::InputArgList &args) {
   return std::nullopt;
 }
 
-static std::unique_ptr<llvm::vfs::FileSystem>
-getVFS(COFFLinkerContext &ctx, const opt::InputArgList &args) {
-  using namespace llvm::vfs;
-
+static void parseVFSOverlay(COFFLinkerContext &ctx,
+                            const opt::InputArgList &args) {
   const opt::Arg *arg = args.getLastArg(OPT_vfsoverlay);
   if (!arg)
-    return nullptr;
-
-  auto bufOrErr = llvm::MemoryBuffer::getFile(arg->getValue());
-  if (!bufOrErr) {
-    checkError(errorCodeToError(bufOrErr.getError()));
-    return nullptr;
-  }
-
-  if (auto ret = vfs::getVFSFromYAML(std::move(*bufOrErr),
-                                     /*DiagHandler*/ nullptr, arg->getValue()))
-    return ret;
+    return;
 
-  Err(ctx) << "Invalid vfs overlay";
-  return nullptr;
+  auto baseFS = ctx.vfs ? ctx.vfs : llvm::vfs::createPhysicalFileSystem();
+  ctx.vfs = lld::createVFSFromOverlay(
+      arg->getValue(), std::move(baseFS),
+      [&](const Twine &msg) { Err(ctx) << msg; });
 }
 
 static StringRef DllDefaultEntryPoint(MachineTypes machine, bool mingw) {
@@ -1612,7 +1605,7 @@ void LinkerDriver::linkerMain(ArrayRef<const char *> argsArr) {
     ctx.e.errorLimit = n;
   }
 
-  config->vfs = getVFS(ctx, args);
+  parseVFSOverlay(ctx, args);
 
   // Handle /help
   if (args.hasArg(OPT_help)) {
diff --git a/lld/COFF/DriverUtils.cpp b/lld/COFF/DriverUtils.cpp
index 42c7f9338151..0ac9260e8d21 100644
--- a/lld/COFF/DriverUtils.cpp
+++ b/lld/COFF/DriverUtils.cpp
@@ -16,6 +16,7 @@
 #include "Driver.h"
 #include "Symbols.h"
 #include "lld/Common/ErrorHandler.h"
+#include "lld/Common/Filesystem.h"
 #include "lld/Common/Memory.h"
 #include "llvm/ADT/StringExtras.h"
 #include "llvm/ADT/StringSwitch.h"
@@ -219,7 +220,7 @@ void LinkerDriver::parseSectionLayout(StringRef path) {
   if (path.starts_with("@"))
     path = path.substr(1);
   std::unique_ptr<MemoryBuffer> layoutFile =
-      CHECK(MemoryBuffer::getFile(path), "could not open " + path);
+      CHECK(lld::readFileVFS(ctx.vfs.get(), path), "could not open " + path);
   StringRef content = layoutFile->getBuffer();
   int index = 0;
 
@@ -253,7 +254,7 @@ void LinkerDriver::parseSectionLayout(StringRef path) {
 
 void LinkerDriver::parseDosStub(StringRef path) {
   std::unique_ptr<MemoryBuffer> stub =
-      CHECK(MemoryBuffer::getFile(path), "could not open " + path);
+      CHECK(lld::readFileVFS(ctx.vfs.get(), path), "could not open " + path);
   size_t bufferSize = stub->getBufferSize();
   const char *bufferStart = stub->getBufferStart();
   // MS link.exe compatibility:
diff --git a/lld/COFF/PDB.cpp b/lld/COFF/PDB.cpp
index fd54d20da3cc..febc0f31204d 100644
--- a/lld/COFF/PDB.cpp
+++ b/lld/COFF/PDB.cpp
@@ -16,6 +16,7 @@
 #include "Symbols.h"
 #include "TypeMerger.h"
 #include "Writer.h"
+#include "lld/Common/Filesystem.h"
 #include "lld/Common/Timer.h"
 #include "llvm/DebugInfo/CodeView/DebugFrameDataSubsection.h"
 #include "llvm/DebugInfo/CodeView/DebugInlineeLinesSubsection.h"
@@ -1304,7 +1305,7 @@ void PDBLinker::addNatvisFiles() {
   llvm::TimeTraceScope timeScope("Natvis files");
   for (StringRef file : ctx.config.natvisFiles) {
     ErrorOr<std::unique_ptr<MemoryBuffer>> dataOrErr =
-        MemoryBuffer::getFile(file);
+        lld::readFileVFS(ctx.vfs.get(), file);
     if (!dataOrErr) {
       Warn(ctx) << "Cannot open input file: " << file;
       continue;
@@ -1326,7 +1327,7 @@ void PDBLinker::addNamedStreams() {
   for (const auto &streamFile : ctx.config.namedStreams) {
     const StringRef stream = streamFile.getKey(), file = streamFile.getValue();
     ErrorOr<std::unique_ptr<MemoryBuffer>> dataOrErr =
-        MemoryBuffer::getFile(file);
+        lld::readFileVFS(ctx.vfs.get(), file);
     if (!dataOrErr) {
       Warn(ctx) << "Cannot open input file: " << file;
       continue;
diff --git a/lld/COFF/SymbolTable.cpp b/lld/COFF/SymbolTable.cpp
index 38a43390c15a..1bd0aff33827 100644
--- a/lld/COFF/SymbolTable.cpp
+++ b/lld/COFF/SymbolTable.cpp
@@ -14,6 +14,7 @@
 #include "PDB.h"
 #include "Symbols.h"
 #include "lld/Common/ErrorHandler.h"
+#include "lld/Common/Filesystem.h"
 #include "lld/Common/Memory.h"
 #include "lld/Common/Timer.h"
 #include "llvm/DebugInfo/DIContext.h"
@@ -1300,9 +1301,8 @@ void SymbolTable::assignExportOrdinals() {
 void SymbolTable::parseModuleDefs(StringRef path) {
   llvm::TimeTraceScope timeScope("Parse def file");
   std::unique_ptr<MemoryBuffer> mb =
-      CHECK(MemoryBuffer::getFile(path, /*IsText=*/false,
-                                  /*RequiresNullTerminator=*/false,
-                                  /*IsVolatile=*/true),
+      CHECK(lld::readFileVFS(ctx.vfs.get(), path, /*isText=*/false,
+                             /*requiresNullTerminator=*/false),
             "could not open " + path);
   COFFModuleDefinition m = check(parseCOFFModuleDefinition(
       mb->getMemBufferRef(), machine, ctx.config.mingw));
diff --git a/lld/Common/DriverDispatcher.cpp b/lld/Common/DriverDispatcher.cpp
index 0b71c0809486..0523e9add145 100644
--- a/lld/Common/DriverDispatcher.cpp
+++ b/lld/Common/DriverDispatcher.cpp
@@ -18,6 +18,7 @@
 #include "llvm/Support/CrashRecoveryContext.h"
 #include "llvm/Support/Path.h"
 #include "llvm/Support/Process.h"
+#include "llvm/Support/VirtualFileSystem.h"
 #include "llvm/TargetParser/Host.h"
 #include "llvm/TargetParser/Triple.h"
 #include <cstdlib>
@@ -144,7 +145,10 @@ static Driver whichDriver(llvm::SmallVectorImpl<const char *> &argsV,
   if (it == drivers.end()) {
     // Driver is invalid or not available in this build.
     return [](llvm::ArrayRef<const char *>, llvm::raw_ostream &,
-              llvm::raw_ostream &, bool, bool) { return false; };
+              llvm::raw_ostream &, bool, bool,
+              llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem>) {
+      return false;
+    };
   }
   return it->d;
 }
@@ -156,11 +160,13 @@ bool inTestOutputDisabled = false;
 /// windows linker based on the argv[0] or -flavor option.
 int unsafeLldMain(llvm::ArrayRef<const char *> args,
                   llvm::raw_ostream &stdoutOS, llvm::raw_ostream &stderrOS,
-                  llvm::ArrayRef<DriverDef> drivers, bool exitEarly) {
+                  llvm::ArrayRef<DriverDef> drivers, bool exitEarly,
+                  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> vfs) {
   SmallVector<const char *, 256> argsV(args);
   Driver d = whichDriver(argsV, drivers);
   // Run the driver. If an error occurs, false will be returned.
-  int r = !d(argsV, stdoutOS, stderrOS, exitEarly, inTestOutputDisabled);
+  int r = !d(argsV, stdoutOS, stderrOS, exitEarly, inTestOutputDisabled,
+             std::move(vfs));
   // At this point 'r' is either 1 for error, and 0 for no error.
 
   // Call exit() if we can to avoid calling destructors.
@@ -177,7 +183,8 @@ int unsafeLldMain(llvm::ArrayRef<const char *> args,
 
 Result lld::lldMain(llvm::ArrayRef<const char *> args,
                     llvm::raw_ostream &stdoutOS, llvm::raw_ostream &stderrOS,
-                    llvm::ArrayRef<DriverDef> drivers) {
+                    llvm::ArrayRef<DriverDef> drivers,
+                    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> vfs) {
   int r = 0;
   {
     // The crash recovery is here only to be able to recover from arbitrary
@@ -186,7 +193,7 @@ Result lld::lldMain(llvm::ArrayRef<const char *> args,
     llvm::CrashRecoveryContext crc;
     if (!crc.RunSafely([&]() {
           r = unsafeLldMain(args, stdoutOS, stderrOS, drivers,
-                            /*exitEarly=*/false);
+                            /*exitEarly=*/false, std::move(vfs));
         }))
       return {crc.RetCode, /*canRunAgain=*/false};
   }
diff --git a/lld/Common/Filesystem.cpp b/lld/Common/Filesystem.cpp
index c2d3644191c9..0a6dd7afbc87 100644
--- a/lld/Common/Filesystem.cpp
+++ b/lld/Common/Filesystem.cpp
@@ -15,9 +15,11 @@
 #include "llvm/Config/llvm-config.h"
 #include "llvm/Support/FileOutputBuffer.h"
 #include "llvm/Support/FileSystem.h"
+#include "llvm/Support/MemoryBuffer.h"
 #include "llvm/Support/Parallel.h"
 #include "llvm/Support/Path.h"
 #include "llvm/Support/TimeProfiler.h"
+#include "llvm/Support/VirtualFileSystem.h"
 #if LLVM_ON_UNIX
 #include <unistd.h>
 #endif
@@ -155,3 +157,47 @@ std::unique_ptr<raw_fd_ostream> lld::openLTOOutputFile(StringRef file) {
     return fs;
   return openFile(file);
 }
+
+ErrorOr<std::unique_ptr<MemoryBuffer>>
+lld::readFileVFS(llvm::vfs::FileSystem *vfs, const Twine &path, bool isText,
+                 bool requiresNullTerminator) {
+  if (!vfs)
+    return MemoryBuffer::getFile(path, isText, requiresNullTerminator);
+
+  auto bufOrErr = vfs->getBufferForFile(path, /*FileSize=*/-1,
+                                        requiresNullTerminator,
+                                        /*IsVolatile=*/false, isText);
+  if (!bufOrErr)
+    return bufOrErr.getError();
+  return std::move(*bufOrErr);
+}
+
+bool lld::existsVFS(llvm::vfs::FileSystem *vfs, const Twine &path) {
+  if (!vfs)
+    return sys::fs::exists(path);
+  return vfs->exists(path);
+}
+
+IntrusiveRefCntPtr<llvm::vfs::FileSystem> lld::createVFSFromOverlay(
+    StringRef overlayPath,
+    IntrusiveRefCntPtr<llvm::vfs::FileSystem> baseFS,
+    function_ref<void(const Twine &)> errHandler) {
+  if (!baseFS)
+    baseFS = llvm::vfs::createPhysicalFileSystem();
+
+  auto bufOrErr = baseFS->getBufferForFile(overlayPath);
+  if (!bufOrErr) {
+    errHandler("cannot open VFS overlay file " + overlayPath + ": " +
+               bufOrErr.getError().message());
+    return baseFS;
+  }
+
+  auto fs = llvm::vfs::getVFSFromYAML(std::move(*bufOrErr),
+                                       /*DiagHandler=*/nullptr, overlayPath,
+                                       /*DiagContext=*/nullptr, baseFS);
+  if (!fs) {
+    errHandler("invalid VFS overlay file " + overlayPath);
+    return baseFS;
+  }
+  return fs;
+}
diff --git a/lld/ELF/Driver.cpp b/lld/ELF/Driver.cpp
index d7bfa7357d4e..37fbf5d9cc66 100644
--- a/lld/ELF/Driver.cpp
+++ b/lld/ELF/Driver.cpp
@@ -64,6 +64,7 @@
 #include "llvm/Support/TarWriter.h"
 #include "llvm/Support/TargetSelect.h"
 #include "llvm/Support/TimeProfiler.h"
+#include "llvm/Support/VirtualFileSystem.h"
 #include "llvm/Support/raw_ostream.h"
 #include <cstdlib>
 #include <tuple>
@@ -116,11 +117,14 @@ llvm::raw_fd_ostream Ctx::openAuxiliaryFile(llvm::StringRef filename,
 namespace lld {
 namespace elf {
 bool link(ArrayRef<const char *> args, llvm::raw_ostream &stdoutOS,
-          llvm::raw_ostream &stderrOS, bool exitEarly, bool disableOutput) {
+          llvm::raw_ostream &stderrOS, bool exitEarly, bool disableOutput,
+          llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> vfs) {
   // This driver-specific context will be freed later by unsafeLldMain().
   auto *context = new Ctx;
   Ctx &ctx = *context;
 
+  ctx.vfs = std::move(vfs);
+
   context->e.initialize(stdoutOS, stderrOS, exitEarly, disableOutput);
   context->e.logName = args::getFilenameWithoutExe(args[0]);
   context->e.errorLimitExceededMsg =
@@ -700,6 +704,15 @@ void LinkerDriver::linkerMain(ArrayRef<const char *> argsArr) {
     }
   }
 
+  // Parse --vfs-overlay option. Layer YAML overlay on top of any VFS provided
+  // by the library caller (or the real filesystem if none was provided).
+  if (auto *arg = args.getLastArg(OPT_vfs_overlay)) {
+    auto baseFS = ctx.vfs ? ctx.vfs : llvm::vfs::createPhysicalFileSystem();
+    ctx.vfs = lld::createVFSFromOverlay(
+        arg->getValue(), std::move(baseFS),
+        [&](const Twine &msg) { ErrAlways(ctx) << msg; });
+  }
+
   readConfigs(ctx, args);
   checkZOptions(ctx, args);
 
diff --git a/lld/ELF/DriverUtils.cpp b/lld/ELF/DriverUtils.cpp
index 6d027c529c19..ee31a8d5da97 100644
--- a/lld/ELF/DriverUtils.cpp
+++ b/lld/ELF/DriverUtils.cpp
@@ -15,6 +15,7 @@
 #include "Config.h"
 #include "Driver.h"
 #include "lld/Common/CommonLinkerContext.h"
+#include "lld/Common/Filesystem.h"
 #include "lld/Common/Reproduce.h"
 #include "llvm/Option/Option.h"
 #include "llvm/Support/CommandLine.h"
@@ -224,7 +225,7 @@ static std::optional<std::string> findFile(Ctx &ctx, StringRef path1,
   else
     path::append(s, path1, path2);
 
-  if (fs::exists(s))
+  if (lld::existsVFS(ctx.vfs.get(), s))
     return std::string(s);
   return std::nullopt;
 }
@@ -263,7 +264,7 @@ std::optional<std::string> elf::searchLibrary(Ctx &ctx, StringRef name) {
 // look for the script in the '-L' search paths. This matches the behaviour of
 // '-T', --version-script=, and linker script INPUT() command in ld.bfd.
 std::optional<std::string> elf::searchScript(Ctx &ctx, StringRef name) {
-  if (fs::exists(name))
+  if (lld::existsVFS(ctx.vfs.get(), name))
     return name.str();
   return findFromSearchPaths(ctx, name);
 }
diff --git a/lld/ELF/InputFiles.cpp b/lld/ELF/InputFiles.cpp
index 176f6937a8d6..4fe8e11a2a1c 100644
--- a/lld/ELF/InputFiles.cpp
+++ b/lld/ELF/InputFiles.cpp
@@ -17,6 +17,7 @@
 #include "SyntheticSections.h"
 #include "Target.h"
 #include "lld/Common/DWARF.h"
+#include "lld/Common/Filesystem.h"
 #include "llvm/ADT/CachedHashString.h"
 #include "llvm/ADT/STLExtras.h"
 #include "llvm/LTO/LTO.h"
@@ -250,8 +251,8 @@ std::optional<MemoryBufferRef> elf::readFile(Ctx &ctx, StringRef path) {
   Log(ctx) << path;
   ctx.arg.dependencyFiles.insert(llvm::CachedHashString(path));
 
-  auto mbOrErr = MemoryBuffer::getFile(path, /*IsText=*/false,
-                                       /*RequiresNullTerminator=*/false);
+  auto mbOrErr = lld::readFileVFS(ctx.vfs.get(), path, /*isText=*/false,
+                                  /*requiresNullTerminator=*/false);
   if (auto ec = mbOrErr.getError()) {
     ErrAlways(ctx) << "cannot open " << path << ": " << ec.message();
     return std::nullopt;
@@ -390,7 +391,7 @@ static void addDependentLibrary(Ctx &ctx, StringRef specifier,
     ctx.driver.addFile(ctx.saver.save(*s), /*withLOption=*/true);
   else if (std::optional<std::string> s = findFromSearchPaths(ctx, specifier))
     ctx.driver.addFile(ctx.saver.save(*s), /*withLOption=*/true);
-  else if (fs::exists(specifier))
+  else if (lld::existsVFS(ctx.vfs.get(), specifier))
     ctx.driver.addFile(specifier, /*withLOption=*/false);
   else
     ErrAlways(ctx)
diff --git a/lld/ELF/LTO.cpp b/lld/ELF/LTO.cpp
index 6f916a501a26..1ea7cca0c729 100644
--- a/lld/ELF/LTO.cpp
+++ b/lld/ELF/LTO.cpp
@@ -72,7 +72,7 @@ static lto::Config createConfig(Ctx &ctx) {
       c.Options.BBSections = BasicBlockSection::None;
     } else {
       ErrorOr<std::unique_ptr<MemoryBuffer>> MBOrErr =
-          MemoryBuffer::getFile(ctx.arg.ltoBasicBlockSections.str());
+          lld::readFileVFS(ctx.vfs.get(), ctx.arg.ltoBasicBlockSections.str());
       if (!MBOrErr) {
         ErrAlways(ctx) << "cannot open " << ctx.arg.ltoBasicBlockSections << ":"
                        << MBOrErr.getError().message();
diff --git a/lld/ELF/Options.td b/lld/ELF/Options.td
index c2111e58c12b..c97b9e5c4ef3 100644
--- a/lld/ELF/Options.td
+++ b/lld/ELF/Options.td
@@ -546,6 +546,8 @@ def power10_stubs: FF<"power10-stubs">, Alias<power10_stubs_eq>, AliasArgs<["yes
 def no_power10_stubs: FF<"no-power10-stubs">, Alias<power10_stubs_eq>, AliasArgs<["no"]>,
   HelpText<"Alias for --power10-stubs=no">;
 
+defm vfs_overlay: EEq<"vfs-overlay", "Path to a YAML VFS overlay file">;
+
 defm version_script: Eq<"version-script", "Read a version script">;
 
 defm warn_backrefs: BB<"warn-backrefs",
diff --git a/lld/ELF/ScriptParser.cpp b/lld/ELF/ScriptParser.cpp
index 0300b956e607..e30f367fbb84 100644
--- a/lld/ELF/ScriptParser.cpp
+++ b/lld/ELF/ScriptParser.cpp
@@ -21,6 +21,7 @@
 #include "SymbolTable.h"
 #include "Symbols.h"
 #include "Target.h"
+#include "lld/Common/Filesystem.h"
 #include "llvm/ADT/SmallString.h"
 #include "llvm/ADT/StringRef.h"
 #include "llvm/ADT/StringSwitch.h"
@@ -316,7 +317,7 @@ void ScriptParser::addFile(StringRef s) {
   if (curBuf.isUnderSysroot && s.starts_with("/")) {
     SmallString<128> pathData;
     StringRef path = (ctx.arg.sysroot + s).toStringRef(pathData);
-    if (sys::fs::exists(path))
+    if (lld::existsVFS(ctx.vfs.get(), path))
       ctx.driver.addFile(ctx.saver.save(path), /*withLOption=*/false);
     else
       setError("cannot find " + s + " inside " + ctx.arg.sysroot);
@@ -343,13 +344,13 @@ void ScriptParser::addFile(StringRef s) {
     if (!directory.empty()) {
       SmallString<0> path(directory);
       sys::path::append(path, s);
-      if (sys::fs::exists(path)) {
+      if (lld::existsVFS(ctx.vfs.get(), path)) {
         ctx.driver.addFile(path, /*withLOption=*/false);
         return;
       }
     }
     // Then search in the current working directory.
-    if (sys::fs::exists(s)) {
+    if (lld::existsVFS(ctx.vfs.get(), s)) {
       ctx.driver.addFile(s, /*withLOption=*/false);
     } else {
       // Finally, search in the list of library paths.
diff --git a/lld/MachO/Driver.cpp b/lld/MachO/Driver.cpp
index 973b3f5535cb..ab6cdb9032a7 100644
--- a/lld/MachO/Driver.cpp
+++ b/lld/MachO/Driver.cpp
@@ -26,6 +26,7 @@
 #include "lld/Common/Args.h"
 #include "lld/Common/CommonLinkerContext.h"
 #include "lld/Common/ErrorHandler.h"
+#include "lld/Common/Filesystem.h"
 #include "lld/Common/LLVM.h"
 #include "lld/Common/Memory.h"
 #include "lld/Common/Reproduce.h"
@@ -50,6 +51,7 @@
 #include "llvm/Support/TargetSelect.h"
 #include "llvm/Support/Threading.h"
 #include "llvm/Support/TimeProfiler.h"
+#include "llvm/Support/VirtualFileSystem.h"
 #include "llvm/TargetParser/Host.h"
 #include "llvm/TextAPI/Architecture.h"
 #include "llvm/TextAPI/PackedVersion.h"
@@ -1710,10 +1712,13 @@ static SmallVector<StringRef, 0> getAllowableClients(opt::InputArgList &args) {
 namespace lld {
 namespace macho {
 bool link(ArrayRef<const char *> argsArr, llvm::raw_ostream &stdoutOS,
-          llvm::raw_ostream &stderrOS, bool exitEarly, bool disableOutput) {
+          llvm::raw_ostream &stderrOS, bool exitEarly, bool disableOutput,
+          llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> vfs) {
   // This driver-specific context will be freed later by lldMain().
   auto *ctx = new CommonLinkerContext;
 
+  ctx->vfs = std::move(vfs);
+
   ctx->e.initialize(stdoutOS, stderrOS, exitEarly, disableOutput);
   ctx->e.cleanupCallback = []() {
     resolvedFrameworks.clear();
@@ -1766,6 +1771,14 @@ bool link(ArrayRef<const char *> argsArr, llvm::raw_ostream &stdoutOS,
     return true;
   }
 
+  // Parse --vfs-overlay option.
+  if (auto *arg = args.getLastArg(OPT_vfs_overlay)) {
+    auto baseFS = ctx->vfs ? ctx->vfs : llvm::vfs::createPhysicalFileSystem();
+    ctx->vfs = lld::createVFSFromOverlay(
+        arg->getValue(), std::move(baseFS),
+        [](const Twine &msg) { error(msg); });
+  }
+
   config = std::make_unique<Configuration>();
   symtab = std::make_unique<SymbolTable>();
   config->outputType = getOutputType(args);
diff --git a/lld/MachO/InputFiles.cpp b/lld/MachO/InputFiles.cpp
index cc7eae51175b..1be529f94e88 100644
--- a/lld/MachO/InputFiles.cpp
+++ b/lld/MachO/InputFiles.cpp
@@ -58,6 +58,7 @@
 
 #include "lld/Common/CommonLinkerContext.h"
 #include "lld/Common/DWARF.h"
+#include "lld/Common/Filesystem.h"
 #include "lld/Common/Reproduce.h"
 #include "llvm/ADT/iterator.h"
 #include "llvm/BinaryFormat/MachO.h"
@@ -218,7 +219,8 @@ std::optional<MemoryBufferRef> macho::readFile(StringRef path) {
     return entry->second;
 
   ErrorOr<std::unique_ptr<MemoryBuffer>> mbOrErr =
-      MemoryBuffer::getFile(path, false, /*RequiresNullTerminator=*/false);
+      lld::readFileVFS(lld::commonContext().vfs.get(), path, /*isText=*/false,
+                       /*requiresNullTerminator=*/false);
   if (std::error_code ec = mbOrErr.getError()) {
     error("cannot open " + path + ": " + ec.message());
     return std::nullopt;
diff --git a/lld/MachO/Options.td b/lld/MachO/Options.td
index 5bd220b3c196..08d4ab4c8fc0 100644
--- a/lld/MachO/Options.td
+++ b/lld/MachO/Options.td
@@ -108,6 +108,12 @@ def thinlto_cache_policy_eq: Joined<["--"], "thinlto-cache-policy=">,
     Group<grp_lld>;
 def O : JoinedOrSeparate<["-"], "O">,
     HelpText<"Optimize output file size">;
+def vfs_overlay: Separate<["--"], "vfs-overlay">,
+    HelpText<"Path to a YAML VFS overlay file">,
+    Group<grp_lld>;
+def vfs_overlay_eq: Joined<["--"], "vfs-overlay=">,
+    Alias<vfs_overlay>,
+    Group<grp_lld>;
 def start_lib: Flag<["--"], "start-lib">,
     HelpText<"Start a grouping of objects that should be treated as if they were together in an archive">;
 def end_lib: Flag<["--"], "end-lib">,
diff --git a/lld/MinGW/Driver.cpp b/lld/MinGW/Driver.cpp
index 70639573b397..0fa98a96779a 100644
--- a/lld/MinGW/Driver.cpp
+++ b/lld/MinGW/Driver.cpp
@@ -41,6 +41,7 @@
 #include "llvm/Support/CommandLine.h"
 #include "llvm/Support/FileSystem.h"
 #include "llvm/Support/Path.h"
+#include "llvm/Support/VirtualFileSystem.h"
 #include "llvm/TargetParser/Host.h"
 #include "llvm/TargetParser/Triple.h"
 #include <optional>
@@ -183,14 +184,16 @@ static bool isI386Target(const opt::InputArgList &args,
 namespace lld {
 namespace coff {
 bool link(ArrayRef<const char *> argsArr, llvm::raw_ostream &stdoutOS,
-          llvm::raw_ostream &stderrOS, bool exitEarly, bool disableOutput);
+          llvm::raw_ostream &stderrOS, bool exitEarly, bool disableOutput,
+          llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> vfs);
 }
 
 namespace mingw {
 // Convert Unix-ish command line arguments to Windows-ish ones and
 // then call coff::link.
 bool link(ArrayRef<const char *> argsArr, llvm::raw_ostream &stdoutOS,
-          llvm::raw_ostream &stderrOS, bool exitEarly, bool disableOutput) {
+          llvm::raw_ostream &stderrOS, bool exitEarly, bool disableOutput,
+          llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> vfs) {
   auto *ctx = new CommonLinkerContext;
   ctx->e.initialize(stdoutOS, stderrOS, exitEarly, disableOutput);
 
@@ -617,7 +620,8 @@ bool link(ArrayRef<const char *> argsArr, llvm::raw_ostream &stdoutOS,
   // The context will be re-created in the COFF driver.
   lld::CommonLinkerContext::destroy();
 
-  return coff::link(vec, stdoutOS, stderrOS, exitEarly, disableOutput);
+  return coff::link(vec, stdoutOS, stderrOS, exitEarly, disableOutput,
+                    std::move(vfs));
 }
 } // namespace mingw
 } // namespace lld
diff --git a/lld/include/lld/Common/CommonLinkerContext.h b/lld/include/lld/Common/CommonLinkerContext.h
index 3641bb70306c..3670b5dbc4f3 100644
--- a/lld/include/lld/Common/CommonLinkerContext.h
+++ b/lld/include/lld/Common/CommonLinkerContext.h
@@ -21,7 +21,9 @@
 
 #include "lld/Common/ErrorHandler.h"
 #include "lld/Common/Memory.h"
+#include "llvm/ADT/IntrusiveRefCntPtr.h"
 #include "llvm/Support/StringSaver.h"
+#include "llvm/Support/VirtualFileSystem.h"
 
 namespace llvm {
 class raw_ostream;
@@ -42,6 +44,10 @@ public:
   llvm::DenseMap<void *, SpecificAllocBase *> instances;
 
   ErrorHandler e;
+
+  // Virtual file system for reading input files. When null, files are read
+  // directly from disk via MemoryBuffer::getFile().
+  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> vfs;
 };
 
 // Retrieve the global state. Currently only one state can exist per process,
diff --git a/lld/include/lld/Common/Driver.h b/lld/include/lld/Common/Driver.h
index 8520e6e7e257..330e3e14a268 100644
--- a/lld/include/lld/Common/Driver.h
+++ b/lld/include/lld/Common/Driver.h
@@ -10,6 +10,8 @@
 #define LLD_COMMON_DRIVER_H
 
 #include "llvm/ADT/ArrayRef.h"
+#include "llvm/ADT/IntrusiveRefCntPtr.h"
+#include "llvm/Support/VirtualFileSystem.h"
 #include "llvm/Support/raw_ostream.h"
 
 namespace lld {
@@ -23,7 +25,8 @@ enum Flavor {
 };
 
 using Driver = bool (*)(llvm::ArrayRef<const char *>, llvm::raw_ostream &,
-                        llvm::raw_ostream &, bool, bool);
+                        llvm::raw_ostream &, bool, bool,
+                        llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem>);
 
 struct DriverDef {
   Flavor f;
@@ -42,7 +45,8 @@ struct Result {
 // properly exit your application and avoid intermittent crashes on exit caused
 // by cleanup.
 Result lldMain(llvm::ArrayRef<const char *> args, llvm::raw_ostream &stdoutOS,
-               llvm::raw_ostream &stderrOS, llvm::ArrayRef<DriverDef> drivers);
+               llvm::raw_ostream &stderrOS, llvm::ArrayRef<DriverDef> drivers,
+               llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> vfs = nullptr);
 } // namespace lld
 
 // With this macro, library users must specify which drivers they use, provide
@@ -52,7 +56,8 @@ Result lldMain(llvm::ArrayRef<const char *> args, llvm::raw_ostream &stdoutOS,
   namespace lld {                                                              \
   namespace name {                                                             \
   bool link(llvm::ArrayRef<const char *> args, llvm::raw_ostream &stdoutOS,    \
-            llvm::raw_ostream &stderrOS, bool exitEarly, bool disableOutput);  \
+            llvm::raw_ostream &stderrOS, bool exitEarly, bool disableOutput,   \
+            llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> vfs = nullptr);    \
   }                                                                            \
   }
 
diff --git a/lld/include/lld/Common/Filesystem.h b/lld/include/lld/Common/Filesystem.h
index 61b32eec2ee7..074bd7ccfff4 100644
--- a/lld/include/lld/Common/Filesystem.h
+++ b/lld/include/lld/Common/Filesystem.h
@@ -10,15 +10,39 @@
 #define LLD_FILESYSTEM_H
 
 #include "lld/Common/LLVM.h"
+#include "llvm/ADT/IntrusiveRefCntPtr.h"
+#include "llvm/Support/MemoryBuffer.h"
 #include "llvm/Support/raw_ostream.h"
 #include <memory>
 #include <system_error>
 
+namespace llvm::vfs {
+class FileSystem;
+} // namespace llvm::vfs
+
 namespace lld {
 void unlinkAsync(StringRef path);
 std::error_code tryCreateFile(StringRef path);
 std::unique_ptr<llvm::raw_fd_ostream> openFile(StringRef file);
 std::unique_ptr<llvm::raw_fd_ostream> openLTOOutputFile(StringRef file);
+
+// Read a file through VFS if available, otherwise fall back to
+// MemoryBuffer::getFile(). When vfs is null, this is equivalent to a direct
+// MemoryBuffer::getFile() call with zero overhead.
+llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
+readFileVFS(llvm::vfs::FileSystem *vfs, const llvm::Twine &path,
+            bool isText = false, bool requiresNullTerminator = false);
+
+// Check file existence through VFS if available, otherwise fall back to
+// sys::fs::exists().
+bool existsVFS(llvm::vfs::FileSystem *vfs, const llvm::Twine &path);
+
+// Create a VFS from a YAML overlay file, layered on top of the given base
+// filesystem. If baseFS is null, the real filesystem is used as base.
+llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem>
+createVFSFromOverlay(llvm::StringRef overlayPath,
+                     llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> baseFS,
+                     llvm::function_ref<void(const llvm::Twine &)> errHandler);
 } // namespace lld
 
 #endif
diff --git a/lld/rfc.md b/lld/rfc.md
new file mode 100644
index 000000000000..14641ec5c35d
--- /dev/null
+++ b/lld/rfc.md
@@ -0,0 +1,297 @@
+# RFC: Virtual File System (VFS) Support in LLD
+
+## Problem Statement
+
+### Background
+
+LLVM has a mature Virtual File System abstraction (`llvm/Support/VirtualFileSystem.h`) that allows tools to virtualize all file I/O. Clang uses it extensively via `FileManager` to support features like VFS overlays, reproducible builds, and in-memory compilation. However, LLD — the LLVM linker — largely bypasses VFS and reads files directly from disk via `MemoryBuffer::getFile()`.
+
+### Motivation: GPU Offloading Pipeline
+
+The primary motivating use case is the GPU offloading pipeline in `clang-linker-wrapper`. Today, this pipeline:
+
+1. Compiles device code to `.o` files, writing them to temporary files on disk
+2. Spawns LLD as a subprocess to link these `.o` files
+3. LLD reads them back from disk
+4. The linked device binary is embedded into the host object
+
+This creates unnecessary disk I/O overhead. With VFS support, the pipeline could instead:
+
+1. Compile device code to in-memory buffers
+2. Call `lldMain()` in-process, passing an `InMemoryFileSystem` containing the compiler outputs
+3. Eliminate all intermediate temporary file creation and disk round-trips
+
+### Current State
+
+| Backend | VFS Status |
+|---------|-----------|
+| **COFF** | Partial — `/vfsoverlay` option exists but VFS is only used for file discovery (`vfs->status()`), not actual reads |
+| **ELF** | None — uses `--chroot` and `--remap-inputs` as path-level alternatives |
+| **Mach-O** | None |
+| **WASM** | None |
+
+The COFF backend's partial support is asymmetric: it uses VFS for `findFile()` path resolution but reads files directly from disk via `MemoryBuffer::getFile()`. This limits its usefulness to scenarios where file paths need redirection but the files themselves must exist on disk.
+
+### Goals
+
+1. **Library API**: Enable callers of `lldMain()` to pass a VFS containing input files, eliminating disk I/O for in-process linking
+2. **Command-line**: Provide `--vfs-overlay` across all backends for YAML-based file redirection
+3. **Backward compatibility**: Zero behavior change when VFS is not used (null VFS falls through to direct disk I/O)
+4. **Consistency**: Unified VFS infrastructure shared across all four backends
+
+### Non-Goals
+
+- **Output VFS**: Output files (`-o`, import libraries, PDB files) continue to use direct disk I/O. Virtualizing outputs is a future phase with different requirements (streaming writes, `FileOutputBuffer`, `raw_fd_ostream`)
+- **Replacing `--chroot`/`--remap-inputs`**: These ELF features operate at the path string level before VFS resolution. They remain useful and complementary
+
+---
+
+## Proposal
+
+### API Changes
+
+#### `lldMain()` Signature
+
+```cpp
+// Before
+Result lldMain(ArrayRef<const char *> args, raw_ostream &stdoutOS,
+               raw_ostream &stderrOS, ArrayRef<DriverDef> drivers);
+
+// After
+Result lldMain(ArrayRef<const char *> args, raw_ostream &stdoutOS,
+               raw_ostream &stderrOS, ArrayRef<DriverDef> drivers,
+               IntrusiveRefCntPtr<vfs::FileSystem> vfs = nullptr);
+```
+
+The default `nullptr` argument preserves source compatibility with all existing callers.
+
+#### Backend `link()` Signatures
+
+All backend `link()` functions gain a VFS parameter:
+
+```cpp
+// Via LLD_HAS_DRIVER macro
+bool link(ArrayRef<const char *> args, raw_ostream &stdoutOS,
+          raw_ostream &stderrOS, bool exitEarly, bool disableOutput,
+          IntrusiveRefCntPtr<vfs::FileSystem> vfs = nullptr);
+```
+
+#### Usage Examples
+
+```cpp
+// Option 1: Default — reads from real disk, fully backward compatible
+lldMain(args, stdout, stderr, drivers);
+
+// Option 2: In-memory filesystem — all inputs from RAM
+auto memFS = makeIntrusiveRefCnt<vfs::InMemoryFileSystem>();
+memFS->addFile("/tmp/input.o", 0, MemoryBuffer::getMemBuffer(objData));
+lldMain(args, stdout, stderr, drivers, memFS);
+
+// Option 3: Overlay — in-memory inputs + real system libraries
+auto overlay = makeIntrusiveRefCnt<vfs::OverlayFileSystem>(
+    vfs::createPhysicalFileSystem());
+overlay->pushOverlay(memFS);
+lldMain(args, stdout, stderr, drivers, overlay);
+```
+
+### Command-Line Interface
+
+All backends gain a `--vfs-overlay` option:
+
+```
+ld.lld --vfs-overlay overlay.yaml -o output input.o
+ld64.lld --vfs-overlay overlay.yaml -o output input.o
+wasm-ld --vfs-overlay overlay.yaml -o output input.o
+lld-link /vfsoverlay:overlay.yaml  (existing, now uses shared infra)
+```
+
+When both a caller-provided VFS and `--vfs-overlay` are present, the YAML overlay is layered on top of the caller's VFS. This matches Clang's behavior.
+
+---
+
+## Design
+
+### Architecture
+
+```
+                    lldMain(args, vfs)
+                           |
+                    DriverDispatcher
+                           |
+              +------------+------------+
+              |            |            |
+          elf::link   coff::link   macho::link  ...
+              |            |            |
+          ctx.vfs = vfs    |            |
+              |            |            |
+         --vfs-overlay?    |            |
+              |            |            |
+    createVFSFromOverlay   |            |
+    (layers on ctx.vfs)    |            |
+              |            |            |
+         readFileVFS()  readFileVFS()  readFileVFS()
+              |            |            |
+         vfs != null?  vfs != null?  vfs != null?
+           /    \        /    \        /    \
+         VFS   disk    VFS   disk    VFS   disk
+```
+
+### VFS Storage
+
+The VFS is stored in `CommonLinkerContext`, the base class for all backend contexts:
+
+```cpp
+class CommonLinkerContext {
+public:
+  // ...existing members...
+  IntrusiveRefCntPtr<vfs::FileSystem> vfs;
+};
+```
+
+- **ELF**: `Ctx` inherits `CommonLinkerContext`, so `ctx.vfs` is directly available
+- **COFF**: `COFFLinkerContext` inherits `CommonLinkerContext`, same pattern
+- **Mach-O**: Uses `CommonLinkerContext` directly (accessed via `commonContext().vfs`)
+- **WASM**: Has its own `Ctx` struct (not inheriting `CommonLinkerContext`), so `vfs` is added as a direct member with explicit reset in `Ctx::reset()`
+
+`IntrusiveRefCntPtr` is used (not `unique_ptr`) because the VFS may be shared across multiple LLD invocations from a library caller. This matches the pattern used throughout LLVM's VFS infrastructure and in Clang's `FileManager`.
+
+### Null VFS Fast Path
+
+When VFS is null (the default), helper functions short-circuit directly to the underlying system calls with zero overhead:
+
+```cpp
+ErrorOr<std::unique_ptr<MemoryBuffer>>
+readFileVFS(vfs::FileSystem *vfs, const Twine &path,
+            bool isText, bool requiresNullTerminator) {
+  if (!vfs)
+    return MemoryBuffer::getFile(path, isText, requiresNullTerminator);
+  return vfs->getBufferForFile(path, ...);
+}
+
+bool existsVFS(vfs::FileSystem *vfs, const Twine &path) {
+  if (!vfs)
+    return sys::fs::exists(path);
+  return vfs->exists(path);
+}
+```
+
+This ensures existing users see no performance regression.
+
+### Path Transform Ordering (ELF)
+
+The ELF backend has existing path transformation features:
+
+1. `--chroot`: Prepends a root directory to absolute paths
+2. `--remap-inputs`: Pattern-based path rewriting
+
+These operate on the path **string** before any file I/O. VFS then resolves the **transformed path**. This ordering is correct and requires no semantic changes:
+
+```
+original path  -->  --chroot  -->  --remap-inputs  -->  VFS resolution
+"/lib/foo.o"       "/sysroot/lib/foo.o"                  vfs->getBufferForFile(...)
+```
+
+### Shared VFS Overlay Parsing
+
+A common `createVFSFromOverlay()` function replaces the COFF-specific `getVFS()`:
+
+```cpp
+IntrusiveRefCntPtr<vfs::FileSystem>
+createVFSFromOverlay(StringRef overlayPath,
+                     IntrusiveRefCntPtr<vfs::FileSystem> baseFS,
+                     function_ref<void(const Twine &)> errHandler);
+```
+
+- Reads the YAML overlay file through the base VFS (not direct disk I/O)
+- Layers the overlay on top of the base filesystem
+- If base is null, uses `createPhysicalFileSystem()` as default
+- Error handling is delegated to backend-specific error reporters via callback
+
+### Thread Safety (COFF)
+
+The COFF backend reads files asynchronously via `std::async`. The VFS is captured by value (ref-counted) in the lambda:
+
+```cpp
+return std::async(strategy, [=, vfs = std::move(vfs)]() {
+  return readFileVFS(vfs.get(), path, ...);
+});
+```
+
+Using `createPhysicalFileSystem()` as the base (rather than `getRealFileSystem()`) ensures each thread has an independent CWD, avoiding races.
+
+### What Is NOT Routed Through VFS
+
+Certain file operations intentionally bypass VFS:
+
+- **Output files**: `-o`, import libraries, PDB files use `FileOutputBuffer`/`raw_fd_ostream`
+- **`--reproduce` tar**: Reads files for archival — uses real filesystem to capture actual file state
+- **Temporary files**: Manifest generation, temp import libraries — locally generated artifacts
+- **`rewritePath()`**: Checks file existence for `--reproduce` metadata
+
+These are either output operations or operations where VFS indirection would be semantically incorrect.
+
+---
+
+## Files Changed
+
+### Common Infrastructure
+| File | Change |
+|------|--------|
+| `lld/include/lld/Common/CommonLinkerContext.h` | Added `vfs` member |
+| `lld/include/lld/Common/Driver.h` | VFS parameter in `Driver`, `lldMain()`, `LLD_HAS_DRIVER` |
+| `lld/Common/DriverDispatcher.cpp` | Forward VFS through dispatch |
+| `lld/include/lld/Common/Filesystem.h` | `readFileVFS()`, `existsVFS()`, `createVFSFromOverlay()` |
+| `lld/Common/Filesystem.cpp` | Implementation of VFS utilities |
+
+### ELF Backend
+| File | Change |
+|------|--------|
+| `lld/ELF/Driver.cpp` | Accept VFS, parse `--vfs-overlay` |
+| `lld/ELF/Options.td` | `--vfs-overlay` option |
+| `lld/ELF/InputFiles.cpp` | `readFile()` and dependent library through VFS |
+| `lld/ELF/DriverUtils.cpp` | `findFile()`, `searchScript()` through VFS |
+| `lld/ELF/LTO.cpp` | BB sections file through VFS |
+| `lld/ELF/ScriptParser.cpp` | Linker script file discovery through VFS |
+
+### COFF Backend
+| File | Change |
+|------|--------|
+| `lld/COFF/Config.h` | Removed `vfs` field (moved to context) |
+| `lld/COFF/Driver.cpp` | Replaced `getVFS()`, VFS in `createFutureForFile`, `findFile` |
+| `lld/COFF/DriverUtils.cpp` | Section layout and DOS stub reads through VFS |
+| `lld/COFF/SymbolTable.cpp` | Module def file through VFS |
+| `lld/COFF/PDB.cpp` | Natvis and named stream files through VFS |
+
+### Mach-O Backend
+| File | Change |
+|------|--------|
+| `lld/MachO/Driver.cpp` | Accept VFS, parse `--vfs-overlay` |
+| `lld/MachO/Options.td` | `--vfs-overlay` option |
+| `lld/MachO/InputFiles.cpp` | `readFile()` through VFS |
+
+### WASM Backend
+| File | Change |
+|------|--------|
+| `lld/wasm/Config.h` | Added `vfs` field to `Ctx` |
+| `lld/wasm/Driver.cpp` | Accept VFS, parse `--vfs-overlay`, reset in `Ctx::reset()` |
+| `lld/wasm/Options.td` | `--vfs-overlay` option |
+| `lld/wasm/InputFiles.cpp` | `readFile()` through VFS |
+
+### MinGW
+| File | Change |
+|------|--------|
+| `lld/MinGW/Driver.cpp` | Accept and forward VFS to `coff::link()` |
+
+### Tests
+| File | Change |
+|------|--------|
+| `lld/test/ELF/vfs-overlay.s` | New: basic overlay, library paths, error cases |
+| `lld/test/COFF/vfsoverlay.test` | Updated error messages for shared infra |
+
+---
+
+## Future Work
+
+1. **`clang-linker-wrapper` integration** (Phase 5): Call `lldMain()` in-process with `InMemoryFileSystem` containing device `.o` files, eliminating temporary disk I/O in the GPU offloading pipeline
+2. **Output VFS**: Virtualize output file writes for fully in-memory linking pipelines
+3. **Additional tests**: VFS interaction with LTO cache, thin archives, linker scripts with `INPUT()` directives
diff --git a/lld/test/COFF/vfsoverlay.test b/lld/test/COFF/vfsoverlay.test
index 5e27d6e67151..c07b19870876 100644
--- a/lld/test/COFF/vfsoverlay.test
+++ b/lld/test/COFF/vfsoverlay.test
@@ -7,12 +7,12 @@
 
 # RUN: not lld-link %S/Inputs/hello64.obj /libpath:/noexist /out:%t.exe /entry:main /defaultlib:notstd64 /vfsoverlay:noexist 2>&1 \
 # RUN:   | FileCheck %s
-# CHECK: error: {{[Nn]}}o such file or directory
+# CHECK: error: cannot open VFS overlay file noexist:
 
 # RUN: echo "invalid yaml" > %t/badoverlay.yaml
 # RUN: not lld-link %S/Inputs/hello64.obj /libpath:/noexist /out:%t.exe /entry:main /defaultlib:notstd64 /vfsoverlay:%t/badoverlay.yaml 2>&1 \
 # RUN:   | FileCheck %s --check-prefix=BAD-OVERLAY
-# BAD-OVERLAY: error: Invalid vfs overlay
+# BAD-OVERLAY: error: invalid VFS overlay file
 
 #--- overlay.yaml.in
 {
diff --git a/lld/test/ELF/vfs-overlay.s b/lld/test/ELF/vfs-overlay.s
new file mode 100644
index 000000000000..6cd5c0e2d381
--- /dev/null
+++ b/lld/test/ELF/vfs-overlay.s
@@ -0,0 +1,74 @@
+# REQUIRES: x86
+# RUN: rm -rf %t
+# RUN: split-file %s %t
+
+## Create a simple object file.
+# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %t/main.s -o %t/main.o
+
+## Test basic --vfs-overlay: redirect /virtual/main.o to the real main.o.
+# RUN: sed -e "s|REPLACE|%/t/main.o|g" %t/overlay.yaml.in > %t/overlay.yaml
+# RUN: ld.lld --vfs-overlay %t/overlay.yaml -o %t/out /virtual/main.o
+
+## Test --vfs-overlay= (joined form).
+# RUN: ld.lld --vfs-overlay=%t/overlay.yaml -o %t/out /virtual/main.o
+
+## Test --vfs-overlay with -L library search paths.
+# RUN: mkdir -p %t/reallib
+# RUN: llvm-ar rc %t/reallib/libfoo.a %t/main.o
+# RUN: sed -e "s|REPLACE|%/t/reallib|g" %t/lib-overlay.yaml.in > %t/lib-overlay.yaml
+# RUN: ld.lld --vfs-overlay %t/lib-overlay.yaml -L /virtuallib -lfoo -o %t/out --entry=_start
+
+## Test error: invalid overlay file.
+# RUN: echo "invalid yaml" > %t/bad.yaml
+# RUN: not ld.lld --vfs-overlay %t/bad.yaml -o %t/out %t/main.o 2>&1 \
+# RUN:   | FileCheck %s --check-prefix=INVALID
+# INVALID: error: invalid VFS overlay file
+
+## Test error: nonexistent overlay file.
+# RUN: not ld.lld --vfs-overlay %t/nonexistent.yaml -o %t/out %t/main.o 2>&1 \
+# RUN:   | FileCheck %s --check-prefix=NOFILE
+# NOFILE: error: cannot open VFS overlay file
+
+## Test backward compatibility: no --vfs-overlay works as before.
+# RUN: ld.lld -o %t/out %t/main.o
+
+#--- main.s
+.globl _start
+_start:
+  ret
+
+#--- overlay.yaml.in
+{
+  'version': 0,
+  'roots' : [
+    {
+      'name': '/virtual',
+      'type': 'directory',
+      'contents': [
+        {
+          'name': 'main.o',
+          'type': 'file',
+          'external-contents': 'REPLACE'
+        }
+      ]
+    }
+  ]
+}
+
+#--- lib-overlay.yaml.in
+{
+  'version': 0,
+  'roots' : [
+    {
+      'name': '/virtuallib',
+      'type': 'directory',
+      'contents': [
+        {
+          'name': 'libfoo.a',
+          'type': 'file',
+          'external-contents': 'REPLACE/libfoo.a'
+        }
+      ]
+    }
+  ]
+}
diff --git a/lld/tools/lld/lld.cpp b/lld/tools/lld/lld.cpp
index d6800fa1eea4..0ae7b7bb4551 100644
--- a/lld/tools/lld/lld.cpp
+++ b/lld/tools/lld/lld.cpp
@@ -36,6 +36,7 @@
 #include "llvm/Support/LLVMDriver.h"
 #include "llvm/Support/Path.h"
 #include "llvm/Support/PluginLoader.h"
+#include "llvm/Support/VirtualFileSystem.h"
 #include "llvm/Support/Process.h"
 #include "llvm/TargetParser/Host.h"
 #include "llvm/TargetParser/Triple.h"
@@ -53,7 +54,8 @@ extern bool inTestOutputDisabled;
 // LLD-as-lib scenarios.
 int unsafeLldMain(llvm::ArrayRef<const char *> args,
                   llvm::raw_ostream &stdoutOS, llvm::raw_ostream &stderrOS,
-                  llvm::ArrayRef<DriverDef> drivers, bool exitEarly);
+                  llvm::ArrayRef<DriverDef> drivers, bool exitEarly,
+                  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> vfs = nullptr);
 } // namespace lld
 
 // When in lit tests, tells how many times the LLD tool should re-execute the
diff --git a/lld/wasm/Config.h b/lld/wasm/Config.h
index 31e08e4e248a..2b625f0a846c 100644
--- a/lld/wasm/Config.h
+++ b/lld/wasm/Config.h
@@ -9,6 +9,7 @@
 #ifndef LLD_WASM_CONFIG_H
 #define LLD_WASM_CONFIG_H
 
+#include "llvm/ADT/IntrusiveRefCntPtr.h"
 #include "llvm/ADT/SmallVector.h"
 #include "llvm/ADT/StringRef.h"
 #include "llvm/ADT/StringSet.h"
@@ -19,6 +20,9 @@
 
 namespace llvm {
 enum class CodeGenOptLevel;
+namespace vfs {
+class FileSystem;
+} // namespace vfs
 } // namespace llvm
 
 namespace lld::wasm {
@@ -138,6 +142,9 @@ struct Config {
 struct Ctx {
   Config arg;
 
+  // Virtual file system for reading input files.
+  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> vfs;
+
   llvm::SmallVector<ObjFile *, 0> objectFiles;
   llvm::SmallVector<StubFile *, 0> stubFiles;
   llvm::SmallVector<SharedFile *, 0> sharedFiles;
diff --git a/lld/wasm/Driver.cpp b/lld/wasm/Driver.cpp
index b1e36f2ecff7..9adb56f15e46 100644
--- a/lld/wasm/Driver.cpp
+++ b/lld/wasm/Driver.cpp
@@ -31,6 +31,7 @@
 #include "llvm/Support/Process.h"
 #include "llvm/Support/TarWriter.h"
 #include "llvm/Support/TargetSelect.h"
+#include "llvm/Support/VirtualFileSystem.h"
 #include "llvm/TargetParser/Host.h"
 #include <optional>
 
@@ -69,6 +70,7 @@ void Ctx::reset() {
   isPic = false;
   legacyFunctionTable = false;
   emitBssSegments = false;
+  vfs = nullptr;
   sym = WasmSym{};
 }
 
@@ -126,10 +128,13 @@ static bool hasZOption(opt::InputArgList &args, StringRef key) {
 } // anonymous namespace
 
 bool link(ArrayRef<const char *> args, llvm::raw_ostream &stdoutOS,
-          llvm::raw_ostream &stderrOS, bool exitEarly, bool disableOutput) {
+          llvm::raw_ostream &stderrOS, bool exitEarly, bool disableOutput,
+          llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> vfs) {
   // This driver-specific context will be freed later by unsafeLldMain().
   auto *context = new CommonLinkerContext;
 
+  ctx.vfs = std::move(vfs);
+
   context->e.initialize(stdoutOS, stderrOS, exitEarly, disableOutput);
   context->e.cleanupCallback = []() { ctx.reset(); };
   context->e.logName = args::getFilenameWithoutExe(args[0]);
@@ -1318,6 +1323,14 @@ void LinkerDriver::linkerMain(ArrayRef<const char *> argsArr) {
   cl::ResetAllOptionOccurrences();
   cl::ParseCommandLineOptions(v.size(), v.data());
 
+  // Parse --vfs-overlay option.
+  if (auto *arg = args.getLastArg(OPT_vfs_overlay)) {
+    auto baseFS = ctx.vfs ? ctx.vfs : llvm::vfs::createPhysicalFileSystem();
+    ctx.vfs = lld::createVFSFromOverlay(
+        arg->getValue(), std::move(baseFS),
+        [](const Twine &msg) { error(msg); });
+  }
+
   readConfigs(args);
   setConfigs();
 
diff --git a/lld/wasm/InputFiles.cpp b/lld/wasm/InputFiles.cpp
index 5e59b3af83f9..8291b18466e4 100644
--- a/lld/wasm/InputFiles.cpp
+++ b/lld/wasm/InputFiles.cpp
@@ -13,6 +13,7 @@
 #include "OutputSegment.h"
 #include "SymbolTable.h"
 #include "lld/Common/CommonLinkerContext.h"
+#include "lld/Common/Filesystem.h"
 #include "lld/Common/Reproduce.h"
 #include "llvm/BinaryFormat/Wasm.h"
 #include "llvm/Object/Binary.h"
@@ -68,7 +69,7 @@ std::unique_ptr<llvm::TarWriter> tar;
 std::optional<MemoryBufferRef> readFile(StringRef path) {
   log("Loading: " + path);
 
-  auto mbOrErr = MemoryBuffer::getFile(path);
+  auto mbOrErr = lld::readFileVFS(ctx.vfs.get(), path);
   if (auto ec = mbOrErr.getError()) {
     error("cannot open " + path + ": " + ec.message());
     return std::nullopt;
diff --git a/lld/wasm/Options.td b/lld/wasm/Options.td
index 33ecf03176d3..77cac0a7cacc 100644
--- a/lld/wasm/Options.td
+++ b/lld/wasm/Options.td
@@ -165,6 +165,8 @@ def verbose: F<"verbose">, HelpText<"Verbose mode">;
 
 def version: F<"version">, HelpText<"Display the version number and exit">;
 
+defm vfs_overlay: EEq<"vfs-overlay", "Path to a YAML VFS overlay file">;
+
 def warn_unresolved_symbols: F<"warn-unresolved-symbols">,
   HelpText<"Report unresolved symbols as warnings">;
 
diff --git a/llvm/vfs.md b/llvm/vfs.md
new file mode 100644
index 000000000000..764d63e26e5a
--- /dev/null
+++ b/llvm/vfs.md
@@ -0,0 +1,181 @@
+# LLVM Virtual File System (VFS) — Explained Simply
+
+## What is a Virtual File System (VFS)?
+
+Think of VFS as a **fake file system that sits between your program and the real disk**. When your program says "open file `/foo/bar.h`", instead of always going to the actual disk, the VFS intercepts the call and can:
+
+1. **Return a file from memory** (never touches disk at all)
+2. **Redirect to a different file on disk** (you ask for `/a/b.h`, it gives you `/x/y.h`)
+3. **Fall through to the real disk** if it doesn't know the file
+
+It's like a receptionist that intercepts your mail — sometimes she hands you a memo from her desk (in-memory), sometimes she reroutes you to a different mailbox (redirect), and sometimes she just says "go check the real mailbox" (pass-through).
+
+---
+
+## The LLVM VFS Class Hierarchy
+
+Everything lives in `llvm/include/llvm/Support/VirtualFileSystem.h` and the implementation in `llvm/lib/Support/VirtualFileSystem.cpp`. The key classes:
+
+```
+FileSystem (abstract base)
+├── RealFileSystem        — Talks to the actual OS disk
+├── InMemoryFileSystem    — Files exist only in RAM
+├── OverlayFileSystem     — Stack of FSes, first match wins
+├── ProxyFileSystem       — Decorator: wraps another FS, overrides selectively
+└── RedirectingFileSystem — YAML-configured path remapping
+```
+
+### What each does
+
+**`RealFileSystem`** — The boring one. It just calls the OS. `open()`, `stat()`, etc. This is what you get if nobody sets up a VFS.
+
+**`InMemoryFileSystem`** — The cool one. You programmatically add files:
+```cpp
+auto MemFS = makeIntrusiveRefCnt<vfs::InMemoryFileSystem>();
+MemFS->addFile("/fake/path/foo.o", timestamp, memoryBuffer);
+```
+Now when someone does `openFileForRead("/fake/path/foo.o")`, they get that buffer back — no disk involved. Internally it builds a tree of `InMemoryDirectory` → `InMemoryFile` nodes, each `InMemoryFile` holding a `MemoryBuffer`.
+
+**`OverlayFileSystem`** — A stack. You push file systems on top of each other:
+```
+┌─────────────────────┐
+│  InMemoryFileSystem  │  ← checked first
+├─────────────────────┤
+│  RealFileSystem      │  ← fallback
+└─────────────────────┘
+```
+When you look up a file, it checks top-to-bottom. First hit wins. This is the key composition mechanism — you overlay fake files on top of the real disk.
+
+**`RedirectingFileSystem`** — Configured via a YAML file. Maps virtual paths to real paths:
+```yaml
+{ 'version': 0,
+  'roots': [
+    { 'type': 'file',
+      'name': '/virtual/input.o',
+      'external-contents': '/real/path/input.o' }
+  ]}
+```
+Has three modes: `fallthrough` (try virtual, then real), `fallback` (try real, then virtual), `redirect-only` (only virtual).
+
+**`ProxyFileSystem`** — Just forwards everything to a wrapped FS. You subclass it and override only the methods you care about.
+
+---
+
+## How Clang Uses It
+
+Clang's `FileManager` (in `clang/include/clang/Basic/FileManager.h`) holds an `IntrusiveRefCntPtr<llvm::vfs::FileSystem>`. Every file access in Clang — opening headers, reading source files, PCH files — goes through this VFS.
+
+The typical setup:
+```cpp
+// Default: just the real disk
+auto FS = vfs::getRealFileSystem();
+
+// Or: overlay an InMemoryFS on top of the real one
+auto Overlay = makeIntrusiveRefCnt<vfs::OverlayFileSystem>(vfs::getRealFileSystem());
+auto MemFS = makeIntrusiveRefCnt<vfs::InMemoryFileSystem>();
+MemFS->addFile(...);
+Overlay->pushOverlay(MemFS);
+
+// Or: load a YAML VFS overlay
+auto RedirectFS = vfs::getVFSFromYAML(yamlBuffer, handler, yamlPath);
+```
+
+This is how Clang can:
+- Do **reproducible builds** (remap dependency paths)
+- Run **tests without writing temp files** to disk
+- Support **module caching** where files are virtualized
+
+---
+
+## Current State of lld
+
+lld has **minimal** VFS support — only in the COFF linker (`lld/COFF/`):
+- It stores a `std::unique_ptr<llvm::vfs::FileSystem> vfs` in its config
+- It supports `--vfs-overlay` to load a YAML overlay
+- But it's superficial — most file I/O in lld still uses raw `llvm::MemoryBuffer::getFile()` or `sys::fs::` calls directly, bypassing any VFS
+
+The ELF and Mach-O linkers have essentially **zero** VFS integration.
+
+---
+
+## What Would "Adding VFS to lld" Mean?
+
+### Step 1: Route all file I/O through VFS
+
+Right now lld does things like:
+```cpp
+// Direct disk access — bypasses VFS
+auto bufOrErr = MemoryBuffer::getFile(path);
+```
+
+This needs to become:
+```cpp
+// Goes through VFS — could be in-memory, redirected, or real
+auto fileOrErr = FS->openFileForRead(path);
+auto bufOrErr = (*fileOrErr)->getBuffer(path);
+```
+
+Every place lld opens a file (input `.o` files, linker scripts, archive libraries, version scripts, etc.) needs to go through the VFS instead of directly to disk.
+
+### Step 2: Thread the VFS through lld's infrastructure
+
+You need to pass the `FileSystem` reference to every component that reads files:
+- The driver (command-line parsing, response files)
+- The input file reader
+- The linker script parser
+- Archive handling
+- Symbol resolution that opens lazy archives
+
+### Step 3: Enable InMemoryFileSystem for the Code Object Manager use case
+
+This is the payoff. Instead of:
+```
+compiler → writes .o to disk → linker reads .o from disk
+```
+
+You get:
+```
+compiler → puts .o in InMemoryFileSystem → linker reads .o from same InMemoryFS
+```
+
+No disk I/O for intermediate files. The compiler and linker share a VFS where object files exist only in RAM.
+
+---
+
+## The Nuances
+
+### Why not just use pipes/stdin?
+Pipes are sequential — you can only read once, and you need to know the order. With a VFS, files have names and can be accessed randomly, just like disk. The linker can open `foo.o`, then `bar.o`, then go back to `foo.o` — impossible with a pipe.
+
+### Why is this in LLVM core and not Clang?
+It **already is** in LLVM core (`llvm/Support/VirtualFileSystem.h`). Clang just uses it. The challenge is that lld mostly ignores it and does raw disk I/O.
+
+### Reference counting matters
+`FileSystem` uses `IntrusiveRefCntPtr` (like a shared_ptr). This means multiple components can share the same VFS without lifetime issues. When the compiler and linker share an `InMemoryFileSystem`, reference counting ensures it stays alive as long as either needs it.
+
+### The OverlayFS pattern is key
+You rarely use `InMemoryFileSystem` alone. The typical pattern:
+
+```
+OverlayFileSystem
+├── InMemoryFileSystem  (compiler outputs live here)
+└── RealFileSystem      (system libraries, linker scripts still on disk)
+```
+
+This way the linker can find compiler-generated `.o` files in memory but still access `/usr/lib/libc.a` from disk.
+
+### UniqueID is how deduplication works
+Each in-memory file gets a unique ID computed from `hash(parent_id, name, contents)`. This is how the system knows two paths point to the same file (like hardlinks). If you add the same content under two names, it still works correctly.
+
+---
+
+## TL;DR
+
+| Concept | Analogy |
+|---|---|
+| `RealFileSystem` | Going to the filing cabinet yourself |
+| `InMemoryFileSystem` | Someone hands you the document from memory |
+| `OverlayFileSystem` | Check the desk first, then the filing cabinet |
+| `RedirectingFileSystem` | A forwarding address ("file X is actually at path Y") |
+| Adding VFS to lld | Replace every `open(path)` in lld with `vfs->open(path)` |
+| The Code Object Manager goal | Compiler and linker share a virtual filing cabinet in RAM — no paper (disk) needed |
