// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(llvm-to-memref-access)" | FileCheck %s

module {
  func.func @plain_cpp(%arg0: !llvm.ptr {enzymexla.memref_type = memref<4xf32>}, %arg1: !llvm.ptr {enzymexla.memref_type = memref<4x5xf32>, llvm.readonly}) {
    %0 = llvm.mlir.constant(0 : i64) : i64
    %1 = llvm.getelementptr inbounds %arg1[%0] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %2 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> f32
    %3 = llvm.getelementptr inbounds %arg0[%0] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %2, %3 {alignment = 4 : i64} : f32, !llvm.ptr
    return
  }
}

// CHECK-LABEL: func.func @plain_cpp
// CHECK-SAME:  %[[OUT:.*]]: memref<4xf32>
// CHECK-SAME:  %[[IN:.*]]: memref<4x5xf32> {llvm.readonly}
// CHECK:       %[[OUT_PTR:.*]] = "enzymexla.memref2pointer"(%[[OUT]]) : (memref<4xf32>) -> !llvm.ptr
// CHECK:       %[[IN_PTR:.*]] = "enzymexla.memref2pointer"(%[[IN]]) : (memref<4x5xf32>) -> !llvm.ptr
// CHECK:       llvm.getelementptr {{.*}} %[[IN_PTR]]
// CHECK:       llvm.getelementptr {{.*}} %[[OUT_PTR]]
