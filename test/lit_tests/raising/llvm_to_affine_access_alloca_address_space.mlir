// RUN: enzymexlamlir-opt %s --llvm-to-affine-access -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @preserve_matching_address_space(
// CHECK:         %[[ALLOCA:.*]] = memref.alloca() : memref<4xi32, 5>
// CHECK-NOT:     llvm.alloca
// CHECK:         memref.copy %arg0, %[[ALLOCA]] : memref<4xi32, 5> to memref<4xi32, 5>
// CHECK:         memref.copy %[[ALLOCA]], %arg1 : memref<4xi32, 5> to memref<4xi32, 5>
func.func @preserve_matching_address_space(
    %source: memref<4xi32, 5>, %destination: memref<4xi32, 5>) {
  %count = llvm.mlir.constant(4 : i64) : i64
  %alloca = llvm.alloca %count x i32 : (i64) -> !llvm.ptr<5>
  %memref = "enzymexla.pointer2memref"(%alloca)
      : (!llvm.ptr<5>) -> memref<4xi32, 5>
  memref.copy %source, %memref : memref<4xi32, 5> to memref<4xi32, 5>
  memref.copy %memref, %destination : memref<4xi32, 5> to memref<4xi32, 5>
  return
}

// -----

// CHECK-LABEL: func.func @reject_address_space_change(
// CHECK:         %[[ALLOCA:.*]] = llvm.alloca %{{.*}} x i32 : (i64) -> !llvm.ptr<5>
// CHECK:         %[[MEMREF:.*]] = "enzymexla.pointer2memref"(%[[ALLOCA]]) : (!llvm.ptr<5>) -> memref<4xi32, 3>
// CHECK-NOT:     memref.alloca
// CHECK:         memref.copy %arg0, %[[MEMREF]] : memref<4xi32, 3> to memref<4xi32, 3>
func.func @reject_address_space_change(
    %source: memref<4xi32, 3>, %destination: memref<4xi32, 3>) {
  %count = llvm.mlir.constant(4 : i64) : i64
  %alloca = llvm.alloca %count x i32 : (i64) -> !llvm.ptr<5>
  %memref = "enzymexla.pointer2memref"(%alloca)
      : (!llvm.ptr<5>) -> memref<4xi32, 3>
  memref.copy %source, %memref : memref<4xi32, 3> to memref<4xi32, 3>
  memref.copy %memref, %destination : memref<4xi32, 3> to memref<4xi32, 3>
  return
}
