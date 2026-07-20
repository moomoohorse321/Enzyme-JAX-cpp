// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(llvm-to-affine-access)" | FileCheck %s

// CHECK-LABEL: func.func @preserve_explicit_alignment(
// CHECK:         %[[ALLOCA:.*]] = memref.alloca() {alignment = 64 : i64} : memref<4xi32>
// CHECK-NOT:     llvm.alloca
// CHECK:         memref.copy %arg0, %[[ALLOCA]] : memref<4xi32> to memref<4xi32>
// CHECK:         memref.copy %[[ALLOCA]], %arg1 : memref<4xi32> to memref<4xi32>
func.func @preserve_explicit_alignment(%source: memref<4xi32>,
                                       %destination: memref<4xi32>) {
  %count = llvm.mlir.constant(4 : i64) : i64
  %alloca = llvm.alloca %count x i32 {alignment = 64 : i64} : (i64) -> !llvm.ptr
  %memref = "enzymexla.pointer2memref"(%alloca) : (!llvm.ptr) -> memref<4xi32>
  memref.copy %source, %memref : memref<4xi32> to memref<4xi32>
  memref.copy %memref, %destination : memref<4xi32> to memref<4xi32>
  return
}

// CHECK-LABEL: func.func @reject_invalid_explicit_alignment(
// CHECK:         %[[ALLOCA:.*]] = llvm.alloca %{{.*}} x i32 {alignment = 3 : i64}
// CHECK:         %[[MEMREF:.*]] = "enzymexla.pointer2memref"(%[[ALLOCA]])
// CHECK-NOT:     memref.alloca
// CHECK:         memref.copy %arg0, %[[MEMREF]] : memref<4xi32> to memref<4xi32>
// CHECK:         memref.copy %[[MEMREF]], %arg1 : memref<4xi32> to memref<4xi32>
func.func @reject_invalid_explicit_alignment(%source: memref<4xi32>,
                                             %destination: memref<4xi32>) {
  %count = llvm.mlir.constant(4 : i64) : i64
  %alloca = llvm.alloca %count x i32 {alignment = 3 : i64} : (i64) -> !llvm.ptr
  %memref = "enzymexla.pointer2memref"(%alloca) : (!llvm.ptr) -> memref<4xi32>
  memref.copy %source, %memref : memref<4xi32> to memref<4xi32>
  memref.copy %memref, %destination : memref<4xi32> to memref<4xi32>
  return
}
