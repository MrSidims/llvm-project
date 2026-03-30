// REQUIRES: amdgpu-registered-target

// Test that -mfallback-arch adds additional GPU architectures for compilation.
// The primary target and fallback targets should both be compiled.

// RUN: %clang -### --target=x86_64-linux-gnu \
// RUN:   --offload-arch=gfx950 \
// RUN:   -mfallback-arch=gfx942 \
// RUN:   -nogpuinc -nogpulib \
// RUN:   %s 2>&1 | FileCheck --check-prefix=SINGLE-FALLBACK %s

// SINGLE-FALLBACK: "-cc1" "-triple" "amdgcn-amd-amdhsa"{{.*}}"-target-cpu" "gfx942"
// SINGLE-FALLBACK: "-cc1" "-triple" "amdgcn-amd-amdhsa"{{.*}}"-target-cpu" "gfx950"

// Test multiple fallback architectures
// RUN: %clang -### --target=x86_64-linux-gnu \
// RUN:   --offload-arch=gfx950 \
// RUN:   -mfallback-arch=gfx942 \
// RUN:   -mfallback-arch=gfx90a \
// RUN:   -nogpuinc -nogpulib \
// RUN:   %s 2>&1 | FileCheck --check-prefix=MULTI-FALLBACK %s

// MULTI-FALLBACK: "-cc1" "-triple" "amdgcn-amd-amdhsa"{{.*}}"-target-cpu" "gfx90a"
// MULTI-FALLBACK: "-cc1" "-triple" "amdgcn-amd-amdhsa"{{.*}}"-target-cpu" "gfx942"
// MULTI-FALLBACK: "-cc1" "-triple" "amdgcn-amd-amdhsa"{{.*}}"-target-cpu" "gfx950"

// Test comma-separated fallback architectures
// RUN: %clang -### --target=x86_64-linux-gnu \
// RUN:   --offload-arch=gfx950 \
// RUN:   -mfallback-arch=gfx942,gfx90a \
// RUN:   -nogpuinc -nogpulib \
// RUN:   %s 2>&1 | FileCheck --check-prefix=COMMA-FALLBACK %s

// COMMA-FALLBACK: "-cc1" "-triple" "amdgcn-amd-amdhsa"{{.*}}"-target-cpu" "gfx90a"
// COMMA-FALLBACK: "-cc1" "-triple" "amdgcn-amd-amdhsa"{{.*}}"-target-cpu" "gfx942"
// COMMA-FALLBACK: "-cc1" "-triple" "amdgcn-amd-amdhsa"{{.*}}"-target-cpu" "gfx950"

// Test that fallback arch without primary arch still works (adds to default)
// RUN: %clang -### --target=x86_64-linux-gnu \
// RUN:   --offload-arch=gfx908 \
// RUN:   -mfallback-arch=gfx906 \
// RUN:   -nogpuinc -nogpulib \
// RUN:   %s 2>&1 | FileCheck --check-prefix=FALLBACK-ONLY %s

// FALLBACK-ONLY: "-cc1" "-triple" "amdgcn-amd-amdhsa"{{.*}}"-target-cpu" "gfx906"
// FALLBACK-ONLY: "-cc1" "-triple" "amdgcn-amd-amdhsa"{{.*}}"-target-cpu" "gfx908"
