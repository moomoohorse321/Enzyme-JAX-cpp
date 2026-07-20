// RUN: enzymexlamlir-opt %s --llvm-to-affine-access | FileCheck %s

module {
  // CHECK-LABEL: llvm.func @rank_mismatch
  // CHECK: %[[ALLOC:.*]] = llvm.alloca
  // CHECK: %[[VIEW:.*]] = "enzymexla.pointer2memref"(%[[ALLOC]]) : (!llvm.ptr) -> memref<2x2xi32>
  // CHECK: memref.load %[[VIEW]][
  llvm.func @rank_mismatch() -> i32 {
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c0 = arith.constant 0 : index
    %alloc = llvm.alloca %c4 x i32 : (i32) -> !llvm.ptr
    %view = "enzymexla.pointer2memref"(%alloc) : (!llvm.ptr) -> memref<2x2xi32>
    %value = memref.load %view[%c0, %c0] : memref<2x2xi32>
    llvm.return %value : i32
  }

  // CHECK-LABEL: llvm.func @extent_mismatch
  // CHECK: %[[ALLOC:.*]] = llvm.alloca
  // CHECK: %[[VIEW:.*]] = "enzymexla.pointer2memref"(%[[ALLOC]]) : (!llvm.ptr) -> memref<3xi32>
  // CHECK: memref.load %[[VIEW]][
  llvm.func @extent_mismatch() -> i32 {
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c0 = arith.constant 0 : index
    %alloc = llvm.alloca %c4 x i32 : (i32) -> !llvm.ptr
    %view = "enzymexla.pointer2memref"(%alloc) : (!llvm.ptr) -> memref<3xi32>
    %value = memref.load %view[%c0] : memref<3xi32>
    llvm.return %value : i32
  }

  // CHECK-LABEL: llvm.func @layout_mismatch
  // CHECK: %[[ALLOC:.*]] = llvm.alloca
  // CHECK: %[[VIEW:.*]] = "enzymexla.pointer2memref"(%[[ALLOC]]) : (!llvm.ptr) -> memref<4xi32, strided<[2]>>
  // CHECK: memref.load %[[VIEW]][
  llvm.func @layout_mismatch() -> i32 {
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c1 = arith.constant 1 : index
    %alloc = llvm.alloca %c4 x i32 : (i32) -> !llvm.ptr
    %view = "enzymexla.pointer2memref"(%alloc) : (!llvm.ptr) -> memref<4xi32, strided<[2]>>
    %value = memref.load %view[%c1] : memref<4xi32, strided<[2]>>
    llvm.return %value : i32
  }

  // CHECK-LABEL: llvm.func @compatible_dynamic
  // CHECK-NOT: llvm.alloca
  // CHECK: %[[ALLOC:.*]] = memref.alloca() : memref<4xi32>
  // CHECK: memref.load %[[ALLOC]][
  llvm.func @compatible_dynamic() -> i32 {
    %c4 = llvm.mlir.constant(4 : i32) : i32
    %c0 = arith.constant 0 : index
    %alloc = llvm.alloca %c4 x i32 : (i32) -> !llvm.ptr
    %view = "enzymexla.pointer2memref"(%alloc) : (!llvm.ptr) -> memref<?xi32>
    %value = memref.load %view[%c0] : memref<?xi32>
    llvm.return %value : i32
  }
}
