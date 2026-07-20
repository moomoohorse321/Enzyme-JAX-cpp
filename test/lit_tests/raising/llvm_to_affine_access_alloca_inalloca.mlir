// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(llvm-to-affine-access)" | FileCheck %s

module {
  func.func @preserve_inalloca() -> memref<1xi32> {
    %one = llvm.mlir.constant(1 : i64) : i64
    %ptr = llvm.alloca inalloca %one x i32 : (i64) -> !llvm.ptr
    %view = "enzymexla.pointer2memref"(%ptr) : (!llvm.ptr) -> memref<1xi32>
    return %view : memref<1xi32>
  }

  func.func @convert_plain_alloca() -> memref<1xi32> {
    %one = llvm.mlir.constant(1 : i64) : i64
    %ptr = llvm.alloca %one x i32 : (i64) -> !llvm.ptr
    %view = "enzymexla.pointer2memref"(%ptr) : (!llvm.ptr) -> memref<1xi32>
    return %view : memref<1xi32>
  }
}

// CHECK-LABEL: func.func @preserve_inalloca
// CHECK:         %[[PTR:.*]] = llvm.alloca inalloca {{.*}} x i32 : (i64) -> !llvm.ptr
// CHECK:         %[[VIEW:.*]] = "enzymexla.pointer2memref"(%[[PTR]]) : (!llvm.ptr) -> memref<1xi32>
// CHECK:         return %[[VIEW]] : memref<1xi32>

// CHECK-LABEL: func.func @convert_plain_alloca
// CHECK-NOT:     llvm.alloca
// CHECK:         %[[ALLOCA:.*]] = memref.alloca() : memref<1xi32>
// CHECK:         return %[[ALLOCA]] : memref<1xi32>
