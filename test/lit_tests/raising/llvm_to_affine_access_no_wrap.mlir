// RUN: enzymexlamlir-opt %s --llvm-to-affine-access --split-input-file | FileCheck %s

// The source i8 add wraps from 127 to -128. Rebuilding it as affine addition
// would instead use 128; the interior base keeps both addresses in bounds.
// CHECK-LABEL: func.func @llvm_add_wrap(
// CHECK-NOT:     affine.load
// CHECK-NOT:     enzymexla.pointer2memref
// CHECK:         llvm.add %{{.*}}, %{{.*}} : i8
// CHECK:         llvm.load
func.func @llvm_add_wrap(%storage: memref<512xi32>) -> i32 {
  %ptr = "enzymexla.memref2pointer"(%storage)
      : (memref<512xi32>) -> !llvm.ptr
  %interior = llvm.getelementptr %ptr[128]
      : (!llvm.ptr) -> !llvm.ptr, i32
  %base = llvm.mlir.constant(127 : i8) : i8
  %init = arith.constant 0 : i32
  %result = affine.for %i = 0 to 2 iter_args(%acc = %init) -> (i32) {
    %i64 = arith.index_cast %i : index to i64
    %i8 = llvm.trunc %i64 overflow<nsw> : i64 to i8
    %wrapped = llvm.add %base, %i8 : i8
    %index = llvm.sext %wrapped : i8 to i64
    %address = llvm.getelementptr %interior[%index]
        : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %value = llvm.load %address : !llvm.ptr -> i32
    %next = llvm.add %acc, %value : i32
    affine.yield %next : i32
  }
  return %result : i32
}

// -----

// The same rule applies to Arith fixed-width multiplication.
// CHECK-LABEL: func.func @arith_mul_wrap(
// CHECK-NOT:     affine.load
// CHECK-NOT:     enzymexla.pointer2memref
// CHECK:         arith.muli %{{.*}}, %{{.*}} : i8
// CHECK:         llvm.load
func.func @arith_mul_wrap(%storage: memref<512xi32>, %factor: i8) -> i32 {
  %ptr = "enzymexla.memref2pointer"(%storage)
      : (memref<512xi32>) -> !llvm.ptr
  %interior = llvm.getelementptr %ptr[256]
      : (!llvm.ptr) -> !llvm.ptr, i32
  %two = arith.constant 2 : i8
  %init = arith.constant 0 : i32
  %result = affine.for %i = 0 to 1 iter_args(%acc = %init) -> (i32) {
    %wrapped = arith.muli %factor, %two : i8
    %index = llvm.sext %wrapped : i8 to i64
    %address = llvm.getelementptr %interior[%index]
        : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %value = llvm.load %address : !llvm.ptr -> i32
    %next = llvm.add %acc, %value : i32
    affine.yield %next : i32
  }
  return %result : i32
}

// -----

// Signed subtraction can wrap too. Defining it inside the affine scope forces
// access recovery either to rebuild it as affine arithmetic or reject it.
// CHECK-LABEL: func.func @llvm_sub_wrap(
// CHECK-NOT:     affine.load
// CHECK-NOT:     enzymexla.pointer2memref
// CHECK:         llvm.sub %{{.*}}, %arg1 : i8
// CHECK:         llvm.load
func.func @llvm_sub_wrap(%storage: memref<512xi32>, %delta: i8) -> i32 {
  %ptr = "enzymexla.memref2pointer"(%storage)
      : (memref<512xi32>) -> !llvm.ptr
  %interior = llvm.getelementptr %ptr[256]
      : (!llvm.ptr) -> !llvm.ptr, i32
  %min = llvm.mlir.constant(-128 : i8) : i8
  %init = llvm.mlir.constant(0 : i32) : i32
  %result = affine.for %i = 0 to 1 iter_args(%acc = %init) -> (i32) {
    %wrapped = llvm.sub %min, %delta : i8
    %index = llvm.sext %wrapped : i8 to i64
    %address = llvm.getelementptr %interior[%index]
        : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %value = llvm.load %address : !llvm.ptr -> i32
    %next = llvm.add %acc, %value : i32
    affine.yield %next : i32
  }
  return %result : i32
}
