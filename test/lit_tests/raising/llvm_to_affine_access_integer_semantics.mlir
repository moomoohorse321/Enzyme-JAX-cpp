// RUN: enzymexlamlir-opt %s --llvm-to-affine-access --split-input-file | FileCheck %s

// LLVM signed division truncates toward zero, whereas affine floordiv rounds
// toward negative infinity. For i = 1 the two in-bounds indices are 8 and 7.
// CHECK-LABEL: func.func @signed_division_rounding(
// CHECK-NOT:     affine.load
// CHECK-NOT:     enzymexla.pointer2memref
// CHECK:         llvm.sdiv
// CHECK-COUNT-2: llvm.load %{{.*}} : !llvm.ptr -> i32
func.func @signed_division_rounding(%storage: memref<512xi32>) -> i32 {
  %ptr = "enzymexla.memref2pointer"(%storage) : (memref<512xi32>) -> !llvm.ptr
  %init = arith.constant 0 : i32
  %zero = llvm.mlir.constant(0 : i64) : i64
  %two = llvm.mlir.constant(2 : i64) : i64
  %eight = llvm.mlir.constant(8 : i64) : i64
  %result = affine.for %i = 0 to 2 iter_args(%acc = %init) -> (i32) {
    %i64 = arith.index_cast %i : index to i64
    %negative = llvm.sub %zero, %i64 overflow<nsw> : i64
    %quotient = llvm.sdiv %negative, %two : i64
    %index = llvm.add %quotient, %eight overflow<nsw> : i64
    %address = llvm.getelementptr %ptr[%index]
        : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %value = llvm.load %address : !llvm.ptr -> i32
    %direct_address = llvm.getelementptr %ptr[%i64]
        : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %direct = llvm.load %direct_address : !llvm.ptr -> i32
    %sum = llvm.add %value, %direct : i32
    %next = llvm.add %acc, %sum : i32
    affine.yield %next : i32
  }
  return %result : i32
}

// -----

// LLVM srem keeps the dividend's sign; affine mod does not have that rule.
// CHECK-LABEL: func.func @signed_remainder_sign(
// CHECK-NOT:     affine.load
// CHECK-NOT:     enzymexla.pointer2memref
// CHECK:         llvm.srem
// CHECK:         llvm.load %{{.*}} : !llvm.ptr -> i32
func.func @signed_remainder_sign(%storage: memref<512xi32>) -> i32 {
  %ptr = "enzymexla.memref2pointer"(%storage) : (memref<512xi32>) -> !llvm.ptr
  %init = arith.constant 0 : i32
  %zero = llvm.mlir.constant(0 : i64) : i64
  %two = llvm.mlir.constant(2 : i64) : i64
  %eight = llvm.mlir.constant(8 : i64) : i64
  %result = affine.for %i = 0 to 2 iter_args(%acc = %init) -> (i32) {
    %i64 = arith.index_cast %i : index to i64
    %negative = llvm.sub %zero, %i64 overflow<nsw> : i64
    %remainder = llvm.srem %negative, %two : i64
    %index = llvm.add %remainder, %eight overflow<nsw> : i64
    %address = llvm.getelementptr %ptr[%index]
        : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %value = llvm.load %address : !llvm.ptr -> i32
    %next = llvm.add %acc, %value : i32
    affine.yield %next : i32
  }
  return %result : i32
}

// -----

// Unsigned division cannot use signed affine floordiv without a range proof.
// At i = 1, i8(-1) / 2 is 127 rather than -1; both offset addresses are valid.
// CHECK-LABEL: func.func @unsigned_division_interpretation(
// CHECK-NOT:     affine.load
// CHECK-NOT:     enzymexla.pointer2memref
// CHECK:         llvm.udiv
// CHECK:         llvm.load %{{.*}} : !llvm.ptr -> i32
func.func @unsigned_division_interpretation(%storage: memref<512xi32>) -> i32 {
  %ptr = "enzymexla.memref2pointer"(%storage) : (memref<512xi32>) -> !llvm.ptr
  %init = arith.constant 0 : i32
  %interior = llvm.getelementptr %ptr[256]
      : (!llvm.ptr) -> !llvm.ptr, i32
  %zero = llvm.mlir.constant(0 : i8) : i8
  %two = llvm.mlir.constant(2 : i8) : i8
  %result = affine.for %i = 0 to 2 iter_args(%acc = %init) -> (i32) {
    %i64 = arith.index_cast %i : index to i64
    %i8 = llvm.trunc %i64 overflow<nsw> : i64 to i8
    %negative = llvm.sub %zero, %i8 overflow<nsw> : i8
    %quotient = llvm.udiv %negative, %two : i8
    %index = llvm.sext %quotient : i8 to i64
    %address = llvm.getelementptr %interior[%index]
        : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %value = llvm.load %address : !llvm.ptr -> i32
    %next = llvm.add %acc, %value : i32
    affine.yield %next : i32
  }
  return %result : i32
}

// -----

// Logical right shift uses an unsigned interpretation; signed affine floor
// division does not. At i = 1 the two offset addresses are 383 and 255.
// CHECK-LABEL: func.func @logical_shift_interpretation(
// CHECK-NOT:     affine.load
// CHECK-NOT:     enzymexla.pointer2memref
// CHECK:         llvm.lshr
// CHECK:         llvm.load %{{.*}} : !llvm.ptr -> i32
func.func @logical_shift_interpretation(%storage: memref<512xi32>) -> i32 {
  %ptr = "enzymexla.memref2pointer"(%storage) : (memref<512xi32>) -> !llvm.ptr
  %init = arith.constant 0 : i32
  %interior = llvm.getelementptr %ptr[256]
      : (!llvm.ptr) -> !llvm.ptr, i32
  %zero = llvm.mlir.constant(0 : i8) : i8
  %one = llvm.mlir.constant(1 : i8) : i8
  %result = affine.for %i = 0 to 2 iter_args(%acc = %init) -> (i32) {
    %i64 = arith.index_cast %i : index to i64
    %i8 = llvm.trunc %i64 overflow<nsw> : i64 to i8
    %negative = llvm.sub %zero, %i8 overflow<nsw> : i8
    %shifted = llvm.lshr %negative, %one : i8
    %index = llvm.sext %shifted : i8 to i64
    %address = llvm.getelementptr %interior[%index]
        : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %value = llvm.load %address : !llvm.ptr -> i32
    %next = llvm.add %acc, %value : i32
    affine.yield %next : i32
  }
  return %result : i32
}

// -----

// Dropping zext changes i8(-1) from 255 to -1. The interior base keeps both
// candidate addresses in bounds.
// CHECK-LABEL: func.func @zero_extension_interpretation(
// CHECK-NOT:     affine.load
// CHECK-NOT:     enzymexla.pointer2memref
// CHECK:         llvm.zext
// CHECK:         llvm.load %{{.*}} : !llvm.ptr -> i32
func.func @zero_extension_interpretation(%storage: memref<512xi32>) -> i32 {
  %ptr = "enzymexla.memref2pointer"(%storage) : (memref<512xi32>) -> !llvm.ptr
  %init = arith.constant 0 : i32
  %interior = llvm.getelementptr %ptr[256]
      : (!llvm.ptr) -> !llvm.ptr, i32
  %zero = llvm.mlir.constant(0 : i8) : i8
  %result = affine.for %i = 0 to 2 iter_args(%acc = %init) -> (i32) {
    %i64 = arith.index_cast %i : index to i64
    %i8 = llvm.trunc %i64 overflow<nsw> : i64 to i8
    %negative = llvm.sub %zero, %i8 overflow<nsw> : i8
    %index = llvm.zext %negative : i8 to i64
    %address = llvm.getelementptr %interior[%index]
        : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %value = llvm.load %address : !llvm.ptr -> i32
    %next = llvm.add %acc, %value : i32
    affine.yield %next : i32
  }
  return %result : i32
}

// -----

// The loop executes once at i = 256. Dropping trunc changes i8(0) back into
// 256; both indices are valid for the source memref.
// CHECK-LABEL: func.func @truncation_wrap(
// CHECK-NOT:     affine.load
// CHECK-NOT:     enzymexla.pointer2memref
// CHECK:         llvm.trunc
// CHECK:         llvm.load %{{.*}} : !llvm.ptr -> i32
func.func @truncation_wrap(%storage: memref<512xi32>) -> i32 {
  %ptr = "enzymexla.memref2pointer"(%storage) : (memref<512xi32>) -> !llvm.ptr
  %init = arith.constant 0 : i32
  %result = affine.for %i = 256 to 257 iter_args(%acc = %init) -> (i32) {
    %i64 = arith.index_cast %i : index to i64
    %narrow = llvm.trunc %i64 : i64 to i8
    %index = llvm.sext %narrow : i8 to i64
    %address = llvm.getelementptr %ptr[%index]
        : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %value = llvm.load %address : !llvm.ptr -> i32
    %next = llvm.add %acc, %value : i32
    affine.yield %next : i32
  }
  return %result : i32
}

// -----

// Exact division, non-negative extension, and no-signed-wrap truncation are
// sufficient proof for the corresponding affine expressions.
// CHECK-LABEL: func.func @proven_integer_semantics(
// CHECK-NOT:     llvm.load
// CHECK-COUNT-3: affine.load
func.func @proven_integer_semantics(%storage: memref<512xi32>) -> i32 {
  %ptr = "enzymexla.memref2pointer"(%storage) : (memref<512xi32>) -> !llvm.ptr
  %init = arith.constant 0 : i32
  %zero = llvm.mlir.constant(0 : i64) : i64
  %two = llvm.mlir.constant(2 : i64) : i64
  %base = llvm.mlir.constant(256 : i64) : i64
  %result = affine.for %i = 0 to 2 iter_args(%acc = %init) -> (i32) {
    %i64 = arith.index_cast %i : index to i64
    %twice = llvm.mul %i64, %two overflow<nsw> : i64
    %negative = llvm.sub %zero, %twice overflow<nsw> : i64
    %quotient = llvm.sdiv exact %negative, %two : i64
    %div_index = llvm.add %base, %quotient overflow<nsw> : i64
    %div_address = llvm.getelementptr %ptr[%div_index]
        : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %from_div = llvm.load %div_address : !llvm.ptr -> i32

    %i8 = llvm.trunc %i64 overflow<nsw> : i64 to i8
    %extended = llvm.zext nneg %i8 : i8 to i64
    %ext_address = llvm.getelementptr %ptr[%extended]
        : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %from_ext = llvm.load %ext_address : !llvm.ptr -> i32

    %truncated = llvm.trunc %i64 overflow<nsw> : i64 to i8
    %trunc_index = llvm.sext %truncated : i8 to i64
    %trunc_address = llvm.getelementptr %ptr[%trunc_index]
        : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %from_trunc = llvm.load %trunc_address : !llvm.ptr -> i32
    %sum0 = llvm.add %from_div, %from_ext : i32
    %sum1 = llvm.add %sum0, %from_trunc : i32
    %next = llvm.add %acc, %sum1 : i32
    affine.yield %next : i32
  }
  return %result : i32
}

// -----

// An unsupported expression that was never approximated is not a semantic
// rejection: keep the exact GEP and use the existing per-access fallback.
// CHECK-LABEL: func.func @generic_failure_stays_local(
// CHECK-NOT:     llvm.load
// CHECK:         affine.load
// CHECK-NOT:     llvm.load
// CHECK:         llvm.and
// CHECK:         "enzymexla.pointer2memref"
// CHECK:         memref.load
// CHECK-NOT:     llvm.load
// CHECK:         return
func.func @generic_failure_stays_local(%storage: memref<8xi32>) -> i32 {
  %ptr = "enzymexla.memref2pointer"(%storage) : (memref<8xi32>) -> !llvm.ptr
  %init = llvm.load %ptr : !llvm.ptr -> i32
  %mask = llvm.mlir.constant(3 : i64) : i64
  %result = affine.for %i = 0 to 4 iter_args(%acc = %init) -> (i32) {
    %i64 = arith.index_cast %i : index to i64
    %masked = llvm.and %i64, %mask : i64
    %address = llvm.getelementptr inbounds %ptr[%masked]
        : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %value = llvm.load %address : !llvm.ptr -> i32
    %next = llvm.add %acc, %value : i32
    affine.yield %next : i32
  }
  return %result : i32
}

// -----

// Affine expressions compute in the MLIR index width, which need not equal
// the LLVM pointer index width. An i64 exact division cannot be rebuilt in a
// 32-bit affine index without first proving that its operand fits.
// CHECK-LABEL: func.func @narrow_affine_index(
// CHECK-NOT:     affine.load
// CHECK-NOT:     enzymexla.pointer2memref
// CHECK:         llvm.sdiv exact
// CHECK:         llvm.load
module attributes {
  dlti.dl_spec = #dlti.dl_spec<
    index = 32 : i64,
    !llvm.ptr = dense<64> : vector<4xi64>>
} {
  func.func @narrow_affine_index(%storage: memref<?xi32>, %base: i64) -> i32 {
    %ptr = "enzymexla.memref2pointer"(%storage)
        : (memref<?xi32>) -> !llvm.ptr
    %init = arith.constant 0 : i32
    %two = llvm.mlir.constant(2 : i64) : i64
    %twiceBase = llvm.mul %base, %two overflow<nsw> : i64
    %result = affine.for %i = 0 to 4 step 2
        iter_args(%acc = %init) -> (i32) {
      %i64 = arith.index_cast %i : index to i64
      %wide = llvm.add %twiceBase, %i64 overflow<nsw> : i64
      %quotient = llvm.sdiv exact %wide, %two : i64
      %address = llvm.getelementptr %ptr[%quotient]
          : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %value = llvm.load %address : !llvm.ptr -> i32
      %next = llvm.add %acc, %value : i32
      affine.yield %next : i32
    }
    return %result : i32
  }
}
