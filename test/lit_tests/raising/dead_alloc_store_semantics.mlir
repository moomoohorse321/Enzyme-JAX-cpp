// RUN: enzymexlamlir-opt %s --llvm-to-affine-access --split-input-file | FileCheck %s

func.func @preserve_volatile_store() {
  %one = llvm.mlir.constant(1 : i64) : i64
  %value = llvm.mlir.constant(42 : i32) : i32
  %slot = llvm.alloca %one x i32 {alignment = 4 : i64} : (i64) -> !llvm.ptr
  llvm.store volatile %value, %slot {alignment = 4 : i64} : i32, !llvm.ptr
  return
}

// CHECK-LABEL: func.func @preserve_volatile_store()
// CHECK: alloca
// CHECK: store
// CHECK-SAME: volatile

// -----

func.func @preserve_atomic_store() {
  %one = llvm.mlir.constant(1 : i64) : i64
  %value = llvm.mlir.constant(42 : i32) : i32
  %slot = llvm.alloca %one x i32 {alignment = 4 : i64} : (i64) -> !llvm.ptr
  llvm.store %value, %slot atomic monotonic {alignment = 4 : i64} : i32, !llvm.ptr
  return
}

// CHECK-LABEL: func.func @preserve_atomic_store()
// CHECK: alloca
// CHECK: store
// CHECK-SAME: {{atomic monotonic|ordering = 2 : i64}}

// -----

func.func @preserve_raised_volatile_store() {
  %slot = memref.alloca() : memref<1xi32>
  %zero = arith.constant 0 : index
  %value = arith.constant 42 : i32
  memref.store %value, %slot[%zero] {volatile_} : memref<1xi32>
  return
}

// CHECK-LABEL: func.func @preserve_raised_volatile_store()
// CHECK: memref.alloca
// CHECK: memref.store
// CHECK-SAME: volatile_

// -----

func.func @preserve_raised_atomic_store() {
  %slot = memref.alloca() : memref<1xi32>
  %zero = arith.constant 0 : index
  %value = arith.constant 42 : i32
  memref.store %value, %slot[%zero] {ordering = 2 : i64} : memref<1xi32>
  return
}

// CHECK-LABEL: func.func @preserve_raised_atomic_store()
// CHECK: memref.alloca
// CHECK: memref.store
// CHECK-SAME: ordering = 2 : i64

// -----

func.func @erase_plain_dead_store() {
  %slot = memref.alloca() : memref<1xi32>
  %zero = arith.constant 0 : index
  %value = arith.constant 42 : i32
  memref.store %value, %slot[%zero] {ordering = 0 : i64} : memref<1xi32>
  return
}

// CHECK-LABEL: func.func @erase_plain_dead_store()
// CHECK-NOT: alloca
// CHECK-NOT: store
// CHECK: return
