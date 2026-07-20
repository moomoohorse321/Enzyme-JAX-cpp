// RUN: enzymexlamlir-opt %s --canonicalize --split-input-file | FileCheck %s

// A byte value other than zero cannot be replaced by typed null values.
// CHECK-LABEL: func.func @nonzero_memset(
// CHECK-NOT:     scf.for
// CHECK:         "llvm.intr.memset"({{.*}}) <{isVolatile = false}>
func.func @nonzero_memset(%dst: memref<4xi32>) {
  %ptr = "enzymexla.memref2pointer"(%dst) : (memref<4xi32>) -> !llvm.ptr
  %value = arith.constant 42 : i8
  %size = arith.constant 16 : i64
  "llvm.intr.memset"(%ptr, %value, %size) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
  return
}

// -----

// Volatile stores cannot be replaced by ordinary memref stores.
// CHECK-LABEL: func.func @volatile_memset(
// CHECK-NOT:     scf.for
// CHECK:         "llvm.intr.memset"({{.*}}) <{isVolatile = true}>
func.func @volatile_memset(%dst: memref<4xi32>) {
  %ptr = "enzymexla.memref2pointer"(%dst) : (memref<4xi32>) -> !llvm.ptr
  %value = arith.constant 0 : i8
  %size = arith.constant 16 : i64
  "llvm.intr.memset"(%ptr, %value, %size) <{isVolatile = true}> : (!llvm.ptr, i8, i64) -> ()
  return
}

// -----

// A forward elementwise loop does not implement overlapping memmove.
// CHECK-LABEL: func.func @preserve_memmove(
// CHECK-NOT:     scf.for
// CHECK:         "llvm.intr.memmove"({{.*}}) <{isVolatile = false}>
func.func @preserve_memmove(%dst: memref<4xi32>, %src: memref<4xi32>) {
  %dst_ptr = "enzymexla.memref2pointer"(%dst) : (memref<4xi32>) -> !llvm.ptr
  %src_ptr = "enzymexla.memref2pointer"(%src) : (memref<4xi32>) -> !llvm.ptr
  %size = arith.constant 16 : i64
  "llvm.intr.memmove"(%dst_ptr, %src_ptr, %size) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
  return
}

// -----

// Volatile reads and writes cannot be replaced by ordinary memref accesses.
// CHECK-LABEL: func.func @volatile_memcpy(
// CHECK-NOT:     scf.for
// CHECK:         "llvm.intr.memcpy"({{.*}}) <{isVolatile = true}>
func.func @volatile_memcpy(%dst: memref<4xi32>, %src: memref<4xi32>) {
  %dst_ptr = "enzymexla.memref2pointer"(%dst) : (memref<4xi32>) -> !llvm.ptr
  %src_ptr = "enzymexla.memref2pointer"(%src) : (memref<4xi32>) -> !llvm.ptr
  %size = arith.constant 16 : i64
  "llvm.intr.memcpy"(%dst_ptr, %src_ptr, %size) <{isVolatile = true}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
  return
}

// -----

// Disable even the safe-looking zero-fill subset because the same ad-hoc
// expansion family crashes on other valid memref shapes and element types.
// CHECK-LABEL: func.func @zero_memset(
// CHECK-NOT:     scf.for
// CHECK:         "llvm.intr.memset"({{.*}}) <{isVolatile = false}>
func.func @zero_memset(%dst: memref<4xi32>) {
  %ptr = "enzymexla.memref2pointer"(%dst) : (memref<4xi32>) -> !llvm.ptr
  %value = arith.constant 0 : i8
  %size = arith.constant 16 : i64
  "llvm.intr.memset"(%ptr, %value, %size) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
  return
}

// -----

// Plain identity-layout memcpy follows the same conservative boundary in this
// correctness PR; the dedicated raising PR will re-enable its proven subset.
// CHECK-LABEL: func.func @plain_memcpy(
// CHECK-NOT:     scf.for
// CHECK:         "llvm.intr.memcpy"({{.*}}) <{isVolatile = false}>
func.func @plain_memcpy(%dst: memref<4xi32>, %src: memref<4xi32>) {
  %dst_ptr = "enzymexla.memref2pointer"(%dst) : (memref<4xi32>) -> !llvm.ptr
  %src_ptr = "enzymexla.memref2pointer"(%src) : (memref<4xi32>) -> !llvm.ptr
  %size = arith.constant 16 : i64
  "llvm.intr.memcpy"(%dst_ptr, %src_ptr, %size) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
  return
}

// -----

// The old expansion interpreted a dynamic trailing dimension as a constant
// loop bound and crashed while constructing the loop nest.
// CHECK-LABEL: func.func @dynamic_trailing_dimension(
// CHECK-NOT:     scf.for
// CHECK:         "llvm.intr.memcpy"({{.*}}) <{isVolatile = false}>
func.func @dynamic_trailing_dimension(%dst: memref<2x?xi32>,
                                      %src: memref<2x?xi32>) {
  %dst_ptr = "enzymexla.memref2pointer"(%dst) : (memref<2x?xi32>) -> !llvm.ptr
  %src_ptr = "enzymexla.memref2pointer"(%src) : (memref<2x?xi32>) -> !llvm.ptr
  %size = arith.constant 8 : i64
  "llvm.intr.memcpy"(%dst_ptr, %src_ptr, %size) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
  return
}

// -----

// Sub-byte elements made the old byte-width calculation divide by zero.
// CHECK-LABEL: func.func @subbyte_element(
// CHECK-NOT:     scf.for
// CHECK:         "llvm.intr.memcpy"({{.*}}) <{isVolatile = false}>
func.func @subbyte_element(%dst: memref<8xi1>, %src: memref<8xi1>) {
  %dst_ptr = "enzymexla.memref2pointer"(%dst) : (memref<8xi1>) -> !llvm.ptr
  %src_ptr = "enzymexla.memref2pointer"(%src) : (memref<8xi1>) -> !llvm.ptr
  %size = arith.constant 1 : i64
  "llvm.intr.memcpy"(%dst_ptr, %src_ptr, %size) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
  return
}

// -----

// Logical element iteration is not equivalent to a physical byte copy for a
// strided memref.
// CHECK-LABEL: func.func @strided_layout(
// CHECK-NOT:     scf.for
// CHECK:         "llvm.intr.memcpy"({{.*}}) <{isVolatile = false}>
func.func @strided_layout(%dst: memref<4xi32, strided<[2]>>,
                          %src: memref<4xi32, strided<[2]>>) {
  %dst_ptr = "enzymexla.memref2pointer"(%dst) : (memref<4xi32, strided<[2]>>) -> !llvm.ptr
  %src_ptr = "enzymexla.memref2pointer"(%src) : (memref<4xi32, strided<[2]>>) -> !llvm.ptr
  %size = arith.constant 16 : i64
  "llvm.intr.memcpy"(%dst_ptr, %src_ptr, %size) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
  return
}
