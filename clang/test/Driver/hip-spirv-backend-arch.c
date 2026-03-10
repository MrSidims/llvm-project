// Test --offload-arch=amdgcnspirv-be uses SPIRV backend without -use-spirv-backend flag

// RUN: %clang --offload-new-driver --target=x86_64-unknown-linux-gnu --offload-arch=amdgcnspirv-be \
// RUN:         -nogpuinc -nogpulib -### -x hip %s \
// RUN: 2>&1 | FileCheck %s --check-prefix=CHECK-SPIRV-BE

// Verify spirv64-amd-amdhsa triple is used and llvm-spirv translator is NOT invoked
// CHECK-SPIRV-BE-NOT: llvm-spirv
// CHECK-SPIRV-BE: "-cc1" "-triple" "spirv64-amd-amdhsa"
// CHECK-SPIRV-BE-NOT: llvm-spirv

// RUN: %clang --offload-new-driver --target=x86_64-unknown-linux-gnu --offload-arch=amdgcnspirv-be \
// RUN:         -nogpuinc -nogpulib -### -x hip %s --offload-device-only \
// RUN: 2>&1 | FileCheck %s --check-prefix=CHECK-DEVICE-ONLY

// CHECK-DEVICE-ONLY-NOT: llvm-spirv
// CHECK-DEVICE-ONLY: "-cc1" "-triple" "spirv64-amd-amdhsa"
// CHECK-DEVICE-ONLY-NOT: llvm-spirv

// RUN: %clang --no-offload-new-driver --target=x86_64-unknown-linux-gnu --offload-arch=amdgcnspirv-be \
// RUN:         -nogpuinc -nogpulib -### -x hip %s \
// RUN: 2>&1 | FileCheck %s --check-prefix=CHECK-OLD-DRIVER

// CHECK-OLD-DRIVER-NOT: llvm-spirv
// CHECK-OLD-DRIVER: "-cc1" "-triple" "spirv64-amd-amdhsa"
// CHECK-OLD-DRIVER-NOT: llvm-spirv
