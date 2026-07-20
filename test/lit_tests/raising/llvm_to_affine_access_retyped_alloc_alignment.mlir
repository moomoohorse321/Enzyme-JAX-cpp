// RUN: enzymexlamlir-opt %s --llvm-to-affine-access --split-input-file | FileCheck %s

module {
  llvm.func @requires_i32_alignment(!llvm.ptr {llvm.align = 4 : i64})

  // An allocation without an explicit alignment still has an alignment
  // compatible with its element type. Materialize that natural alignment when
  // retyping i32 storage to i8 and a derived pointer escapes to an align-4
  // callee.
  // CHECK-LABEL: func.func @preserve_natural_alignment_on_escape
  // CHECK-NOT: memref<1xi32>
  // CHECK: %[[ALLOCA:.*]] = memref.alloca() {alignment = 4 : i64} : memref<4xi8>
  // CHECK: %[[SUBVIEW:.*]] = memref.subview %[[ALLOCA]]
  // CHECK: %[[ESCAPED:.*]] = "enzymexla.memref2pointer"(%[[SUBVIEW]])
  // CHECK: llvm.call @requires_i32_alignment(%[[ESCAPED]])
  func.func @preserve_natural_alignment_on_escape() {
    %slot = memref.alloca() : memref<1xi32>
    %raw = "enzymexla.memref2pointer"(%slot)
      : (memref<1xi32>) -> !llvm.ptr
    %bytes = "enzymexla.pointer2memref"(%raw)
      : (!llvm.ptr) -> memref<4xi8>
    %subview = memref.subview %bytes[0] [4] [1]
      : memref<4xi8> to memref<4xi8, strided<[1], offset: 0>>
    %escaped = "enzymexla.memref2pointer"(%subview)
      : (memref<4xi8, strided<[1], offset: 0>>) -> !llvm.ptr
    llvm.call @requires_i32_alignment(%escaped) : (!llvm.ptr) -> ()
    %zero = arith.constant 0 : index
    %value = arith.constant 7 : i8
    memref.store %value, %bytes[%zero] : memref<4xi8>
    return
  }
}

// -----

// An explicit alignment is copied to the replacement allocation, so retyping
// remains safe and useful.
// CHECK-LABEL: func.func @retain_explicit_alignment
// CHECK-NOT: enzymexla.memref2pointer
// CHECK-NOT: enzymexla.pointer2memref
// CHECK: %[[ALLOCA:.*]] = memref.alloca() {alignment = 4 : i64} : memref<4xi8>
// CHECK: memref.copy %[[ALLOCA]], %arg0
func.func @retain_explicit_alignment(%out: memref<4xi8>) {
  %slot = memref.alloca() {alignment = 4 : i64} : memref<1xi32>
  %raw = "enzymexla.memref2pointer"(%slot)
    : (memref<1xi32>) -> !llvm.ptr
  %bytes = "enzymexla.pointer2memref"(%raw)
    : (!llvm.ptr) -> memref<4xi8>
  memref.copy %bytes, %out : memref<4xi8> to memref<4xi8>
  return
}
