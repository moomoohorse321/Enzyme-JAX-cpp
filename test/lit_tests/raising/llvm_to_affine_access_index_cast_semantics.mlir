// RUN: enzymexlamlir-opt %s --llvm-to-affine-access --split-input-file | FileCheck %s

// The loop executes once at i = 256. A signed index -> i8 cast produces zero;
// erasing it would instead access element 256. Both source addresses are valid.
// CHECK-LABEL: func.func @signed_narrowing(
// CHECK-NOT:     affine.load
// CHECK-NOT:     enzymexla.pointer2memref
// CHECK:         llvm.load
// CHECK:         arith.index_cast %{{.*}} : index to i8
// CHECK:         llvm.load
func.func @signed_narrowing(%storage: memref<512xi32>) -> i32 {
  %ptr = "enzymexla.memref2pointer"(%storage) : (memref<512xi32>) -> !llvm.ptr
  %init = llvm.load %ptr : !llvm.ptr -> i32
  %result = affine.for %i = 256 to 257 iter_args(%acc = %init) -> (i32) {
    %i8 = arith.index_cast %i : index to i8
    %index = llvm.sext %i8 : i8 to i64
    %address = llvm.getelementptr %ptr[%index]
        : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %value = llvm.load %address : !llvm.ptr -> i32
    %next = llvm.add %acc, %value : i32
    affine.yield %next : i32
  }
  return %result : i32
}

// -----

// Unsigned narrowing loses the same high bits and requires a range proof too.
// CHECK-LABEL: func.func @unsigned_narrowing(
// CHECK-NOT:     affine.load
// CHECK-NOT:     enzymexla.pointer2memref
// CHECK:         arith.index_castui %{{.*}} : index to i8
// CHECK:         llvm.load
func.func @unsigned_narrowing(%storage: memref<512xi32>) -> i32 {
  %ptr = "enzymexla.memref2pointer"(%storage) : (memref<512xi32>) -> !llvm.ptr
  %init = arith.constant 0 : i32
  %result = affine.for %i = 256 to 257 iter_args(%acc = %init) -> (i32) {
    %i8 = arith.index_castui %i : index to i8
    // Keep the following extension independently semantics-preserving so the
    // rejected operation in this case is specifically index_castui narrowing.
    %index = llvm.zext nneg %i8 : i8 to i64
    %address = llvm.getelementptr %ptr[%index]
        : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %value = llvm.load %address : !llvm.ptr -> i32
    %next = llvm.add %acc, %value : i32
    affine.yield %next : i32
  }
  return %result : i32
}

// -----

// Equal-width casts preserve the bit pattern. Signed widening and unsigned
// widening with nneg preserve the mathematical integer value too.
// CHECK-LABEL: func.func @proven_casts(
// CHECK-NOT:     llvm.load
// CHECK-COUNT-3: affine.load
func.func @proven_casts(%storage: memref<512xi32>, %signed: i8,
                        %unsigned: i8) -> i32 {
  %ptr = "enzymexla.memref2pointer"(%storage) : (memref<512xi32>) -> !llvm.ptr
  %center = llvm.getelementptr %ptr[256]
      : (!llvm.ptr) -> !llvm.ptr, i32
  %init = arith.constant 0 : i32
  %result = affine.for %i = 0 to 2 iter_args(%acc = %init) -> (i32) {
    %equalWidth = arith.index_castui %i : index to i64
    %equalAddress = llvm.getelementptr %ptr[%equalWidth]
        : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %equalValue = llvm.load %equalAddress : !llvm.ptr -> i32

    %signedIndex = arith.index_cast %signed : i8 to index
    %signedI64 = arith.index_cast %signedIndex : index to i64
    %signedAddress = llvm.getelementptr %center[%signedI64]
        : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %signedValue = llvm.load %signedAddress : !llvm.ptr -> i32

    %unsignedIndex = arith.index_castui %unsigned nneg : i8 to index
    %unsignedI64 = arith.index_cast %unsignedIndex : index to i64
    %unsignedAddress = llvm.getelementptr %ptr[%unsignedI64]
        : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %unsignedValue = llvm.load %unsignedAddress : !llvm.ptr -> i32

    %sum0 = llvm.add %acc, %equalValue : i32
    %sum1 = llvm.add %sum0, %signedValue : i32
    %sum2 = llvm.add %sum1, %unsignedValue : i32
    affine.yield %sum2 : i32
  }
  return %result : i32
}

// -----

// Cast width follows the MLIR index data layout, not the LLVM pointer width.
// With a 32-bit index, unsigned widening of -1 produces 4294967295; replacing
// it by the original affine induction variable would instead use -1.
// CHECK-LABEL: func.func @index32_unsigned_widening(
// CHECK-NOT:     affine.load
// CHECK-NOT:     enzymexla.pointer2memref
// CHECK:         arith.index_castui %{{.*}} : index to i64
// CHECK:         llvm.load
module attributes {
  dlti.dl_spec = #dlti.dl_spec<
    index = 32 : i64,
    !llvm.ptr = dense<64> : vector<4xi64>>
} {
  func.func @index32_unsigned_widening(%storage: memref<?xi32>) -> i32 {
    %ptr = "enzymexla.memref2pointer"(%storage)
        : (memref<?xi32>) -> !llvm.ptr
    %init = arith.constant 0 : i32
    %result = affine.for %i = -1 to 0 iter_args(%acc = %init) -> (i32) {
      %index = arith.index_castui %i : index to i64
      %address = llvm.getelementptr %ptr[%index]
          : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %value = llvm.load %address : !llvm.ptr -> i32
      %next = llvm.add %acc, %value : i32
      affine.yield %next : i32
    }
    return %result : i32
  }

  // Signed 32-to-64 widening preserves the integer value and remains affine.
  // CHECK-LABEL: func.func @index32_signed_widening(
  // CHECK-NOT:     llvm.load
  // CHECK:         affine.load
  func.func @index32_signed_widening(%storage: memref<?xi32>) -> i32 {
    %ptr = "enzymexla.memref2pointer"(%storage)
        : (memref<?xi32>) -> !llvm.ptr
    %init = arith.constant 0 : i32
    %result = affine.for %i = -1 to 0 iter_args(%acc = %init) -> (i32) {
      %index = arith.index_cast %i : index to i64
      %address = llvm.getelementptr %ptr[%index]
          : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %value = llvm.load %address : !llvm.ptr -> i32
      %next = llvm.add %acc, %value : i32
      affine.yield %next : i32
    }
    return %result : i32
  }
}
