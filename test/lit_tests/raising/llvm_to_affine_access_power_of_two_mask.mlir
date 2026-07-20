// RUN: enzymexlamlir-opt %s --llvm-to-affine-access --split-input-file | FileCheck %s --check-prefix=RAW
// RUN: enzymexlamlir-opt %s --llvm-to-affine-access --canonicalize --split-input-file | FileCheck %s --check-prefix=RAISE

// Clang lowers data[index & 7] to this LLVM and/getelementptr sequence.  A
// low-bit mask is exactly Euclidean modulo 8, including for negative indices.
// RAISE-LABEL: func.func @llvm_masked_store_load(
// RAISE-NOT:     llvm.and
// RAISE-NOT:     mod 8
// RAISE:         affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<?xi32>
// RAISE:         %{{.*}} = affine.load %{{.*}}[%{{.*}}] : memref<?xi32>
// RAW-LABEL: func.func @llvm_masked_store_load(
// RAW:           affine.store %{{.*}}, %{{.*}}[%{{.*}} mod 8]
// RAW:           affine.load %{{.*}}[%{{.*}} mod 8]
func.func @llvm_masked_store_load(%storage: memref<8xi32>) -> i32 {
  %ptr = "enzymexla.memref2pointer"(%storage)
      : (memref<8xi32>) -> !llvm.ptr
  %zero = arith.constant 0 : i32
  %mask = llvm.mlir.constant(7 : i64) : i64
  %result = affine.for %i = 0 to 8 iter_args(%acc = %zero) -> (i32) {
    %index = arith.index_cast %i : index to i64
    %masked = llvm.and %index, %mask : i64
    %address = llvm.getelementptr inbounds %ptr[%masked]
        : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.store %acc, %address : i32, !llvm.ptr
    %loaded = llvm.load %address : !llvm.ptr -> i32
    affine.yield %loaded : i32
  }
  return %result : i32
}

// -----

// The same expression can already be in the Arith dialect by the time access
// recovery runs.
// RAISE-LABEL: func.func @arith_masked_load(
// RAISE-NOT:     arith.andi
// RAISE:         affine.load %{{.*}}[%{{.*}} mod 16] {{.*}}: memref<?xi32>
func.func @arith_masked_load(%storage: memref<16xi32>) -> i32 {
  %ptr = "enzymexla.memref2pointer"(%storage)
      : (memref<16xi32>) -> !llvm.ptr
  %mask = arith.constant 15 : i64
  %init = arith.constant 0 : i32
  %result = affine.for %i = -8 to 8 iter_args(%acc = %init) -> (i32) {
    %index = arith.index_cast %i : index to i64
    %masked = arith.andi %mask, %index : i64
    %address = llvm.getelementptr inbounds %ptr[%masked]
        : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %loaded = llvm.load %address : !llvm.ptr -> i32
    %next = arith.addi %acc, %loaded : i32
    affine.yield %next : i32
  }
  return %result : i32
}

// -----

// A constant with holes is not a modulo mask.  Preserve its exact pointer
// computation and use the existing per-access fallback.
// RAW-LABEL: func.func @non_mask_and_stays_exact(
// RAW:           llvm.and
// RAW:           "enzymexla.pointer2memref"
// RAW:           memref.load
// RAW-NOT:       affine.load
// RAISE-LABEL: func.func @non_mask_and_stays_exact(
// RAISE:         llvm.and
// RAISE:         memref.load
// RAISE-NOT:     affine.load
func.func @non_mask_and_stays_exact(%storage: memref<8xi32>) -> i32 {
  %ptr = "enzymexla.memref2pointer"(%storage)
      : (memref<8xi32>) -> !llvm.ptr
  %mask = llvm.mlir.constant(5 : i64) : i64
  %init = arith.constant 0 : i32
  %result = affine.for %i = 0 to 4 iter_args(%acc = %init) -> (i32) {
    %index = arith.index_cast %i : index to i64
    %masked = llvm.and %index, %mask : i64
    %address = llvm.getelementptr inbounds %ptr[%masked]
        : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %loaded = llvm.load %address : !llvm.ptr -> i32
    %next = llvm.add %acc, %loaded : i32
    affine.yield %next : i32
  }
  return %result : i32
}

// -----

// The modulo must fit the signed range of the target index type.  Otherwise
// lowering an affine.apply could wrap its positive divisor.
// RAISE-LABEL: func.func @mask_too_wide_for_index(
// RAISE:         llvm.and
// RAISE:         memref.load
// RAISE-NOT:     affine.load
// RAW-LABEL: func.func @mask_too_wide_for_index(
// RAW:           llvm.and
// RAW:           memref.load
// RAW-NOT:       affine.load
module attributes {
  dlti.dl_spec = #dlti.dl_spec<
    index = 32 : i64,
    !llvm.ptr = dense<64> : vector<4xi64>>
} {
  func.func @mask_too_wide_for_index(%storage: memref<?xi32>) -> i32 {
    %ptr = "enzymexla.memref2pointer"(%storage)
        : (memref<?xi32>) -> !llvm.ptr
    %mask = llvm.mlir.constant(2147483647 : i64) : i64
    %init = arith.constant 0 : i32
    %result = affine.for %i = 0 to 1 iter_args(%acc = %init) -> (i32) {
      %index = arith.index_cast %i : index to i64
      %masked = llvm.and %index, %mask : i64
      %address = llvm.getelementptr inbounds %ptr[%masked]
          : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %loaded = llvm.load %address : !llvm.ptr -> i32
      %next = llvm.add %acc, %loaded : i32
      affine.yield %next : i32
    }
    return %result : i32
  }
}
