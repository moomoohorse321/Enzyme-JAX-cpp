// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(llvm-to-affine-access)" | FileCheck %s

module attributes {
  dlti.dl_spec = #dlti.dl_spec<
    !llvm.ptr = dense<[64, 64, 64, 64]> : vector<4xi64>,
    !llvm.ptr<1> = dense<[64, 64, 64, 32]> : vector<4xi64>
  >
} {
  // LLVM truncates the i64 GEP index to the address-space-1 pointer's i32
  // index type. Keep that GEP as the address of the conservative memref
  // fallback until the affine builder can model the value-dependent truncation.
  // CHECK-LABEL: func.func @narrow_pointer_index(
  // CHECK-NOT:     affine.load
  // CHECK-NOT:     arith.index_cast
  // CHECK:         %[[ZERO:.*]] = arith.constant 0 : index
  // CHECK:         %[[ADDRESS:.*]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i64
  // CHECK:         %[[MEMREF:.*]] = "enzymexla.pointer2memref"(%[[ADDRESS]]) : (!llvm.ptr<1>) -> memref<?xi64, 1 : index>
  // CHECK:         %[[VALUE:.*]] = memref.load %[[MEMREF]][%[[ZERO]]]
  // CHECK:         return %[[VALUE]] : i64
  func.func @narrow_pointer_index(%storage: memref<8xi64, 1>, %index: i64) -> i64 {
    %ptr = "enzymexla.memref2pointer"(%storage) : (memref<8xi64, 1>) -> !llvm.ptr<1>
    %address = llvm.getelementptr %ptr[%index] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i64
    %value = llvm.load %address : !llvm.ptr<1> -> i64
    return %value : i64
  }

  // An i64 GEP index needs no implicit width conversion for the default
  // address space, whose pointer index type is also i64.
  // CHECK-LABEL: func.func @matching_pointer_index(
  // CHECK-NOT:     llvm.load
  // CHECK:         %[[INDEX:.*]] = arith.index_cast %{{.*}} : i64 to index
  // CHECK:         %[[MEMREF:.*]] = "enzymexla.pointer2memref"(%{{.*}}) : (!llvm.ptr) -> memref<?xi64>
  // CHECK:         %[[VALUE:.*]] = affine.load %[[MEMREF]][symbol(%[[INDEX]])]
  // CHECK:         return %[[VALUE]] : i64
  func.func @matching_pointer_index(%storage: memref<8xi64>, %index: i64) -> i64 {
    %ptr = "enzymexla.memref2pointer"(%storage) : (memref<8xi64>) -> !llvm.ptr
    %address = llvm.getelementptr %ptr[%index] : (!llvm.ptr, i64) -> !llvm.ptr, i64
    %value = llvm.load %address : !llvm.ptr -> i64
    return %value : i64
  }

  // A narrower GEP index is sign-extended to the pointer index width. The
  // signed arith.index_cast materialization preserves that value.
  // CHECK-LABEL: func.func @narrower_gep_index(
  // CHECK-NOT:     llvm.load
  // CHECK:         %[[INDEX:.*]] = arith.index_cast %{{.*}} : i16 to index
  // CHECK:         %[[MEMREF:.*]] = "enzymexla.pointer2memref"(%{{.*}}) : (!llvm.ptr<1>) -> memref<?xi64, 1>
  // CHECK:         %[[VALUE:.*]] = affine.load %[[MEMREF]][symbol(%[[INDEX]])]
  // CHECK:         return %[[VALUE]] : i64
  func.func @narrower_gep_index(%storage: memref<8xi64, 1>, %index: i16) -> i64 {
    %ptr = "enzymexla.memref2pointer"(%storage) : (memref<8xi64, 1>) -> !llvm.ptr<1>
    %address = llvm.getelementptr %ptr[%index] : (!llvm.ptr<1>, i16) -> !llvm.ptr<1>, i64
    %value = llvm.load %address : !llvm.ptr<1> -> i64
    return %value : i64
  }
}
