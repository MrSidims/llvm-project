# RUN: rm -rf %t
# RUN: split-file %s %t

## Create a simple object file.
# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %t/main.s -o %t/main.o

## Test basic --vfs-overlay: redirect /virtual/main.o to the real main.o.
# RUN: sed -e "s|REPLACE|%/t/main.o|g" %t/overlay.yaml.in > %t/overlay.yaml
# RUN: wasm-ld --vfs-overlay %t/overlay.yaml --no-entry --export-all -o %t/out.wasm /virtual/main.o

## Test --vfs-overlay= (joined form).
# RUN: wasm-ld --vfs-overlay=%t/overlay.yaml --no-entry --export-all -o %t/out.wasm /virtual/main.o

## Test --vfs-overlay with -L library search paths.
# RUN: mkdir -p %t/reallib
# RUN: llvm-ar rc %t/reallib/libfoo.a %t/main.o
# RUN: sed -e "s|REPLACE|%/t/reallib|g" %t/lib-overlay.yaml.in > %t/lib-overlay.yaml
# RUN: wasm-ld --vfs-overlay %t/lib-overlay.yaml -L /virtuallib -lfoo --no-entry --export-all -o %t/out.wasm

## Test error: invalid overlay file.
# RUN: echo "invalid yaml" > %t/bad.yaml
# RUN: not wasm-ld --vfs-overlay %t/bad.yaml --no-entry -o %t/out.wasm %t/main.o 2>&1 \
# RUN:   | FileCheck %s --check-prefix=INVALID
# INVALID: error: invalid VFS overlay file

## Test error: nonexistent overlay file.
# RUN: not wasm-ld --vfs-overlay %t/nonexistent.yaml --no-entry -o %t/out.wasm %t/main.o 2>&1 \
# RUN:   | FileCheck %s --check-prefix=NOFILE
# NOFILE: error: cannot open VFS overlay file

## Test backward compatibility: no --vfs-overlay works as before.
# RUN: wasm-ld --no-entry --export-all -o %t/out.wasm %t/main.o

#--- main.s
.globl _start
_start:
  end_function

#--- overlay.yaml.in
{
  'version': 0,
  'roots' : [
    {
      'name': '/virtual',
      'type': 'directory',
      'contents': [
        {
          'name': 'main.o',
          'type': 'file',
          'external-contents': 'REPLACE'
        }
      ]
    }
  ]
}

#--- lib-overlay.yaml.in
{
  'version': 0,
  'roots' : [
    {
      'name': '/virtuallib',
      'type': 'directory',
      'contents': [
        {
          'name': 'libfoo.a',
          'type': 'file',
          'external-contents': 'REPLACE/libfoo.a'
        }
      ]
    }
  ]
}
