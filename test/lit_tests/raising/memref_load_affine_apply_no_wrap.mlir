// RUN: enzymexlamlir-opt %s --llvm-to-affine-access --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @preserve_wrapping_add(
// CHECK:         %[[INDEX:.*]] = arith.addi %{{.*}}, %arg1 : index
// CHECK-NOT:     affine.apply
// CHECK:         memref.load %arg0[%[[INDEX]]]
func.func @preserve_wrapping_add(%storage: memref<?xi32>, %offset: index) -> i32 {
  %init = arith.constant 0 : i32
  %result = affine.for %i = 0 to 1 iter_args(%acc = %init) -> (i32) {
    %index = arith.addi %i, %offset : index
    %value = memref.load %storage[%index] : memref<?xi32>
    %next = arith.addi %acc, %value : i32
    affine.yield %next : i32
  }
  return %result : i32
}

// -----

// CHECK-LABEL: func.func @preserve_wrapping_sub(
// CHECK:         %[[INDEX:.*]] = arith.subi %{{.*}}, %arg1 : index
// CHECK-NOT:     affine.apply
// CHECK:         memref.load %arg0[%[[INDEX]]]
func.func @preserve_wrapping_sub(%storage: memref<?xi32>, %offset: index) -> i32 {
  %init = arith.constant 0 : i32
  %result = affine.for %i = 0 to 1 iter_args(%acc = %init) -> (i32) {
    %index = arith.subi %i, %offset : index
    %value = memref.load %storage[%index] : memref<?xi32>
    %next = arith.addi %acc, %value : i32
    affine.yield %next : i32
  }
  return %result : i32
}

// -----

// CHECK-LABEL: func.func @decompose_nsw_add(
// CHECK:         %[[INDEX:.*]] = affine.apply
// CHECK:         memref.load %arg0[%[[INDEX]]]
func.func @decompose_nsw_add(%storage: memref<?xi32>, %offset: index) -> i32 {
  %init = arith.constant 0 : i32
  %result = affine.for %i = 0 to 1 iter_args(%acc = %init) -> (i32) {
    %index = arith.addi %i, %offset overflow<nsw> : index
    %value = memref.load %storage[%index] : memref<?xi32>
    %next = arith.addi %acc, %value : i32
    affine.yield %next : i32
  }
  return %result : i32
}
