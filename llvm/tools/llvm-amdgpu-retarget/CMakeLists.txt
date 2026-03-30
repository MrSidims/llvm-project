set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsDescs
  AllTargetsDisassemblers
  AllTargetsInfos
  BinaryFormat
  MC
  MCDisassembler
  Object
  Support
  TargetParser
)

add_llvm_tool(llvm-amdgpu-retarget
  llvm-amdgpu-retarget.cpp
  AMDGPURetargeter.cpp
  ELFRetargetWriter.cpp
)
