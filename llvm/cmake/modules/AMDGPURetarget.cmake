#===------------------------------------------------------------------------===#
# AMDGPURetarget.cmake - CMake module for AMDGPU binary retargeting
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===------------------------------------------------------------------------===#
#
# This module provides functions for retargeting AMDGPU code objects at build
# time. It enables cross-generation GPU compatibility by creating versions
# of kernels for multiple GPU architectures from a single source.
#
# Usage:
#   include(AMDGPURetarget)
#
#   # Find the retargeting tool
#   find_amdgpu_retarget()
#
#   # Retarget a single code object
#   amdgpu_retarget_code_object(
#     INPUT ${CMAKE_BINARY_DIR}/kernels/gfx950/kernel.co
#     OUTPUT ${CMAKE_BINARY_DIR}/kernels/gfx942/kernel.co
#     SOURCE_ARCH gfx950
#     TARGET_ARCH gfx942
#   )
#
#   # Retarget all code objects in a directory
#   amdgpu_retarget_directory(
#     SOURCE_DIR ${CMAKE_BINARY_DIR}/kernels/gfx950
#     OUTPUT_DIR ${CMAKE_BINARY_DIR}/kernels/gfx942
#     SOURCE_ARCH gfx950
#     TARGET_ARCH gfx942
#   )
#
#===------------------------------------------------------------------------===#

include_guard(GLOBAL)

# Find the llvm-amdgpu-retarget executable
macro(find_amdgpu_retarget)
  find_program(AMDGPU_RETARGET_EXECUTABLE
    NAMES llvm-amdgpu-retarget
    HINTS
      ${LLVM_TOOLS_BINARY_DIR}
      ${LLVM_BINARY_DIR}/bin
      ${ROCM_PATH}/bin
      /opt/rocm/bin
      /opt/rocm/llvm/bin
    DOC "Path to llvm-amdgpu-retarget executable"
  )

  if(AMDGPU_RETARGET_EXECUTABLE)
    set(AMDGPU_RETARGET_FOUND TRUE)
    message(STATUS "Found llvm-amdgpu-retarget: ${AMDGPU_RETARGET_EXECUTABLE}")
  else()
    set(AMDGPU_RETARGET_FOUND FALSE)
    message(WARNING "llvm-amdgpu-retarget not found; cross-generation kernel "
                    "retargeting will not be available. Build LLVM with "
                    "llvm-amdgpu-retarget enabled or set AMDGPU_RETARGET_EXECUTABLE.")
  endif()
endmacro()

# Retarget a single code object from one architecture to another
#
# Arguments:
#   INPUT         - Path to input code object (.co file)
#   OUTPUT        - Path to output code object
#   SOURCE_ARCH   - Source GPU architecture (e.g., gfx950)
#   TARGET_ARCH   - Target GPU architecture (e.g., gfx942)
#   VERBOSE       - (Optional) Enable verbose output
#
function(amdgpu_retarget_code_object)
  cmake_parse_arguments(ARG
    "VERBOSE"                          # Options
    "INPUT;OUTPUT;SOURCE_ARCH;TARGET_ARCH"  # One-value arguments
    ""                                 # Multi-value arguments
    ${ARGN}
  )

  # Validate required arguments
  if(NOT ARG_INPUT)
    message(FATAL_ERROR "amdgpu_retarget_code_object: INPUT is required")
  endif()
  if(NOT ARG_OUTPUT)
    message(FATAL_ERROR "amdgpu_retarget_code_object: OUTPUT is required")
  endif()
  if(NOT ARG_SOURCE_ARCH)
    message(FATAL_ERROR "amdgpu_retarget_code_object: SOURCE_ARCH is required")
  endif()
  if(NOT ARG_TARGET_ARCH)
    message(FATAL_ERROR "amdgpu_retarget_code_object: TARGET_ARCH is required")
  endif()

  # Check if the retarget tool is available
  if(NOT AMDGPU_RETARGET_FOUND)
    message(FATAL_ERROR "amdgpu_retarget_code_object: llvm-amdgpu-retarget not found. "
                        "Call find_amdgpu_retarget() first.")
  endif()

  # Build command arguments
  set(CMD_ARGS
    "--source=${ARG_SOURCE_ARCH}"
    "--target=${ARG_TARGET_ARCH}"
    "${ARG_INPUT}"
    "-o" "${ARG_OUTPUT}"
  )

  if(ARG_VERBOSE)
    list(APPEND CMD_ARGS "-v")
  endif()

  # Get filename for comment
  get_filename_component(INPUT_NAME "${ARG_INPUT}" NAME)

  # Create custom command
  add_custom_command(
    OUTPUT "${ARG_OUTPUT}"
    COMMAND ${AMDGPU_RETARGET_EXECUTABLE} ${CMD_ARGS}
    DEPENDS "${ARG_INPUT}"
    COMMENT "Retargeting ${INPUT_NAME} from ${ARG_SOURCE_ARCH} to ${ARG_TARGET_ARCH}"
    VERBATIM
  )
endfunction()

# Retarget all code objects in a directory
#
# Arguments:
#   SOURCE_DIR    - Directory containing source code objects
#   OUTPUT_DIR    - Directory for output code objects
#   SOURCE_ARCH   - Source GPU architecture
#   TARGET_ARCH   - Target GPU architecture
#   PATTERN       - (Optional) Glob pattern for files (default: "*.co")
#   TARGET_NAME   - (Optional) Name for the generated target
#   VERBOSE       - (Optional) Enable verbose output
#
function(amdgpu_retarget_directory)
  cmake_parse_arguments(ARG
    "VERBOSE"                                      # Options
    "SOURCE_DIR;OUTPUT_DIR;SOURCE_ARCH;TARGET_ARCH;PATTERN;TARGET_NAME"  # One-value arguments
    ""                                             # Multi-value arguments
    ${ARGN}
  )

  # Validate required arguments
  if(NOT ARG_SOURCE_DIR)
    message(FATAL_ERROR "amdgpu_retarget_directory: SOURCE_DIR is required")
  endif()
  if(NOT ARG_OUTPUT_DIR)
    message(FATAL_ERROR "amdgpu_retarget_directory: OUTPUT_DIR is required")
  endif()
  if(NOT ARG_SOURCE_ARCH)
    message(FATAL_ERROR "amdgpu_retarget_directory: SOURCE_ARCH is required")
  endif()
  if(NOT ARG_TARGET_ARCH)
    message(FATAL_ERROR "amdgpu_retarget_directory: TARGET_ARCH is required")
  endif()

  # Default pattern
  if(NOT ARG_PATTERN)
    set(ARG_PATTERN "*.co")
  endif()

  # Default target name
  if(NOT ARG_TARGET_NAME)
    set(ARG_TARGET_NAME "retarget_${ARG_TARGET_ARCH}_kernels")
  endif()

  # Find all matching files
  file(GLOB CO_FILES "${ARG_SOURCE_DIR}/${ARG_PATTERN}")

  if(NOT CO_FILES)
    message(WARNING "amdgpu_retarget_directory: No files matching ${ARG_PATTERN} "
                    "found in ${ARG_SOURCE_DIR}")
    return()
  endif()

  # Ensure output directory exists
  file(MAKE_DIRECTORY "${ARG_OUTPUT_DIR}")

  # Process each file
  set(RETARGETED_FILES "")
  foreach(CO_FILE ${CO_FILES})
    get_filename_component(CO_NAME "${CO_FILE}" NAME)
    set(OUTPUT_FILE "${ARG_OUTPUT_DIR}/${CO_NAME}")

    amdgpu_retarget_code_object(
      INPUT "${CO_FILE}"
      OUTPUT "${OUTPUT_FILE}"
      SOURCE_ARCH "${ARG_SOURCE_ARCH}"
      TARGET_ARCH "${ARG_TARGET_ARCH}"
      ${ARG_VERBOSE}
    )

    list(APPEND RETARGETED_FILES "${OUTPUT_FILE}")
  endforeach()

  # Create a target that depends on all retargeted files
  add_custom_target(${ARG_TARGET_NAME} ALL
    DEPENDS ${RETARGETED_FILES}
    COMMENT "Retargeting ${ARG_SOURCE_ARCH} kernels to ${ARG_TARGET_ARCH}"
  )

  # Export the list of retargeted files for parent scopes
  set(${ARG_TARGET_NAME}_FILES ${RETARGETED_FILES} PARENT_SCOPE)
endfunction()

# Define fallback architecture chains for common GPU families
#
# This function sets up the standard fallback chains:
#   gfx950 -> gfx942 -> gfx90a (MI300/MI200 family)
#   gfx1100 -> gfx1030 (RDNA3/RDNA2 family)
#
function(amdgpu_setup_fallback_chains)
  set(AMDGPU_FALLBACK_CHAIN_gfx950 "gfx942;gfx90a" CACHE STRING
      "Fallback chain for gfx950 kernels")
  set(AMDGPU_FALLBACK_CHAIN_gfx942 "gfx90a" CACHE STRING
      "Fallback chain for gfx942 kernels")
  set(AMDGPU_FALLBACK_CHAIN_gfx1100 "gfx1030" CACHE STRING
      "Fallback chain for gfx1100 kernels")
endfunction()

# Retarget kernels for all architectures in a fallback chain
#
# Arguments:
#   SOURCE_DIR    - Directory containing source code objects
#   OUTPUT_BASE   - Base directory for output (arch subdirs created)
#   SOURCE_ARCH   - Source GPU architecture
#
function(amdgpu_retarget_with_fallback_chain)
  cmake_parse_arguments(ARG
    "VERBOSE"
    "SOURCE_DIR;OUTPUT_BASE;SOURCE_ARCH"
    ""
    ${ARGN}
  )

  # Get the fallback chain for this architecture
  set(FALLBACK_VAR "AMDGPU_FALLBACK_CHAIN_${ARG_SOURCE_ARCH}")
  if(NOT DEFINED ${FALLBACK_VAR})
    message(WARNING "No fallback chain defined for ${ARG_SOURCE_ARCH}")
    return()
  endif()

  set(FALLBACK_ARCHS "${${FALLBACK_VAR}}")

  # Retarget for each architecture in the chain
  foreach(TARGET_ARCH ${FALLBACK_ARCHS})
    set(OUTPUT_DIR "${ARG_OUTPUT_BASE}/${TARGET_ARCH}")

    amdgpu_retarget_directory(
      SOURCE_DIR "${ARG_SOURCE_DIR}"
      OUTPUT_DIR "${OUTPUT_DIR}"
      SOURCE_ARCH "${ARG_SOURCE_ARCH}"
      TARGET_ARCH "${TARGET_ARCH}"
      TARGET_NAME "retarget_${ARG_SOURCE_ARCH}_to_${TARGET_ARCH}"
      ${ARG_VERBOSE}
    )
  endforeach()
endfunction()
