// Test that --offload-arch=amdgcnspirv selects SPIRVOpenMPToolChain for OpenMP.

// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp \
// RUN:   --offload-arch=amdgcnspirv -nogpulib %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK-SPIRV-TC

// Verify that the spirv64-amd-amdhsa triple is used for the device compilation.
// CHECK-SPIRV-TC: "-cc1" "-triple" "spirv64-amd-amdhsa"
// CHECK-SPIRV-TC-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"

// Verify the host compilation still uses the host triple.
// CHECK-SPIRV-TC: "-cc1" "-triple" "x86_64-unknown-linux-gnu"

// Verify clang-linker-wrapper is invoked.
// CHECK-SPIRV-TC: clang-linker-wrapper

// Test with -fopenmp-targets directly specifying the spirv64-amd-amdhsa triple.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -fopenmp=libomp \
// RUN:   -fopenmp-targets=spirv64-amd-amdhsa -nogpulib %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK-EXPLICIT-TRIPLE

// CHECK-EXPLICIT-TRIPLE: "-cc1" "-triple" "spirv64-amd-amdhsa"
// CHECK-EXPLICIT-TRIPLE-SAME: "-aux-triple" "x86_64-unknown-linux-gnu"

// Verify phases for amdgcnspirv OpenMP offloading.
// RUN: %clang -ccc-print-phases --target=x86_64-unknown-linux-gnu -fopenmp=libomp \
// RUN:   --offload-arch=amdgcnspirv %s 2>&1 \
// RUN: | FileCheck --check-prefix=CHECK-PHASES %s

// CHECK-PHASES: 0: input, "[[INPUT:.+]]", c, (host-openmp)
// CHECK-PHASES: 1: preprocessor, {0}, cpp-output, (host-openmp)
// CHECK-PHASES: 2: compiler, {1}, ir, (host-openmp)
// CHECK-PHASES: 3: input, "[[INPUT]]", c, (device-openmp, amdgcnspirv)
// CHECK-PHASES: 4: preprocessor, {3}, cpp-output, (device-openmp, amdgcnspirv)
// CHECK-PHASES: 5: compiler, {4}, ir, (device-openmp, amdgcnspirv)
// CHECK-PHASES: 6: offload, "host-openmp (x86_64-unknown-linux-gnu)" {2}, "device-openmp (spirv64-amd-amdhsa:amdgcnspirv)" {5}, ir
// CHECK-PHASES: 7: backend, {6}, assembler, (device-openmp, amdgcnspirv)
// CHECK-PHASES: 8: assembler, {7}, object, (device-openmp, amdgcnspirv)

// Verify bindings.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu -ccc-print-bindings \
// RUN:   -fopenmp=libomp --offload-arch=amdgcnspirv -nogpulib %s 2>&1 \
// RUN: | FileCheck %s --check-prefix=CHECK-BINDINGS

// CHECK-BINDINGS: "x86_64-unknown-linux-gnu" - "clang", inputs: ["[[INPUT:.+]]"], output: "[[HOST_BC:.+]]"
// CHECK-BINDINGS: "spirv64-amd-amdhsa" - "clang", inputs: ["[[INPUT]]", "[[HOST_BC]]"], output: "[[DEVICE_SPV:.+]]"
