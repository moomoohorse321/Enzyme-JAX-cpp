// RUN: enzymexlamlir-opt %s --delinearize-indexing --verify-each --split-input-file | FileCheck %s

// An incompatible element type must not be replaced by the argument contract.
// CHECK-LABEL: func.func @element_type_mismatch(
// CHECK:         %[[PTR:.*]] = "enzymexla.memref2pointer"(%arg0) : (memref<16xi32, 1>) -> !llvm.ptr<1>
// CHECK:         %[[VIEW:.*]] = "enzymexla.pointer2memref"(%[[PTR]]) : (!llvm.ptr<1>) -> memref<?xf32, 1>
// CHECK:         %[[VALUE:.*]] = affine.load %[[VIEW]][0] : memref<?xf32, 1>
// CHECK:         memref.store %[[VALUE]], %arg1[] : memref<f32>
func.func @element_type_mismatch(%arg0: memref<16xi32, 1>,
                                 %arg1: memref<f32>) {
  %ptr = "enzymexla.memref2pointer"(%arg0) : (memref<16xi32, 1>) -> !llvm.ptr<1>
  %view = "enzymexla.pointer2memref"(%ptr) : (!llvm.ptr<1>) -> memref<?xf32, 1>
  %value = affine.load %view[0] : memref<?xf32, 1>
  memref.store %value, %arg1[] : memref<f32>
  return
}

// -----

// A compatible linear view still recovers the ranked argument contract.
// CHECK-LABEL: func.func @compatible(
// CHECK:         %[[PTR:.*]] = "enzymexla.memref2pointer"(%arg0) : (memref<16xf32, 1>) -> !llvm.ptr<1>
// CHECK:         %[[VIEW:.*]] = "enzymexla.pointer2memref"(%[[PTR]]) : (!llvm.ptr<1>) -> memref<16xf32, 1>
// CHECK:         %[[VALUE:.*]] = affine.load %[[VIEW]][0] : memref<16xf32, 1>
// CHECK:         return %[[VALUE]] : f32
func.func @compatible(%arg0: memref<16xf32, 1>) -> f32 {
  %ptr = "enzymexla.memref2pointer"(%arg0) : (memref<16xf32, 1>) -> !llvm.ptr<1>
  %view = "enzymexla.pointer2memref"(%ptr) : (!llvm.ptr<1>) -> memref<?xf32, 1>
  %value = affine.load %view[0] : memref<?xf32, 1>
  return %value : f32
}

// -----

// Delinearization consumes one linear index; a ranked access view is not one.
// CHECK-LABEL: func.func @ranked_access_view(
// CHECK:         %[[PTR:.*]] = "enzymexla.memref2pointer"(%arg0) : (memref<4x4xf32, 1>) -> !llvm.ptr<1>
// CHECK:         %[[VIEW:.*]] = "enzymexla.pointer2memref"(%[[PTR]]) : (!llvm.ptr<1>) -> memref<4x4xf32, 1>
// CHECK:         %[[VALUE:.*]] = affine.load %[[VIEW]][%arg1, %arg2] : memref<4x4xf32, 1>
// CHECK:         return %[[VALUE]] : f32
func.func @ranked_access_view(%arg0: memref<4x4xf32, 1>, %arg1: index,
                              %arg2: index) -> f32 {
  %ptr = "enzymexla.memref2pointer"(%arg0) : (memref<4x4xf32, 1>) -> !llvm.ptr<1>
  %view = "enzymexla.pointer2memref"(%ptr) : (!llvm.ptr<1>) -> memref<4x4xf32, 1>
  %value = affine.load %view[%arg1, %arg2] : memref<4x4xf32, 1>
  return %value : f32
}

// -----

// A different memory space is not the same storage contract.
// CHECK-LABEL: func.func @memory_space_mismatch(
// CHECK:         %[[PTR:.*]] = "enzymexla.memref2pointer"(%arg0) : (memref<16xf32, 1>) -> !llvm.ptr<1>
// CHECK:         %[[VIEW:.*]] = "enzymexla.pointer2memref"(%[[PTR]]) : (!llvm.ptr<1>) -> memref<?xf32, 2>
// CHECK:         %[[VALUE:.*]] = affine.load %[[VIEW]][0] : memref<?xf32, 2>
// CHECK:         return %[[VALUE]] : f32
func.func @memory_space_mismatch(%arg0: memref<16xf32, 1>) -> f32 {
  %ptr = "enzymexla.memref2pointer"(%arg0) : (memref<16xf32, 1>) -> !llvm.ptr<1>
  %view = "enzymexla.pointer2memref"(%ptr) : (!llvm.ptr<1>) -> memref<?xf32, 2>
  %value = affine.load %view[0] : memref<?xf32, 2>
  return %value : f32
}

// -----

// Shape-based delinearization is not valid for a non-contiguous contract.
// CHECK-LABEL: func.func @layout_mismatch(
// CHECK:         %[[PTR:.*]] = "enzymexla.memref2pointer"(%arg0) : (memref<4x4xf32, strided<[8, 1]>, 1>) -> !llvm.ptr<1>
// CHECK:         %[[VIEW:.*]] = "enzymexla.pointer2memref"(%[[PTR]]) : (!llvm.ptr<1>) -> memref<?xf32, 1>
// CHECK:         %[[VALUE:.*]] = affine.load %[[VIEW]][%arg1] : memref<?xf32, 1>
// CHECK:         return %[[VALUE]] : f32
func.func @layout_mismatch(
    %arg0: memref<4x4xf32, strided<[8, 1]>, 1>, %arg1: index) -> f32 {
  %ptr = "enzymexla.memref2pointer"(%arg0) : (memref<4x4xf32, strided<[8, 1]>, 1>) -> !llvm.ptr<1>
  %view = "enzymexla.pointer2memref"(%ptr) : (!llvm.ptr<1>) -> memref<?xf32, 1>
  %value = affine.load %view[%arg1] : memref<?xf32, 1>
  return %value : f32
}

// -----

// A non-unit-stride access view is not a linear indexing representation.
// CHECK-LABEL: func.func @access_layout_mismatch(
// CHECK:         %[[PTR:.*]] = "enzymexla.memref2pointer"(%arg0) : (memref<16xf32, 1>) -> !llvm.ptr<1>
// CHECK:         %[[VIEW:.*]] = "enzymexla.pointer2memref"(%[[PTR]]) : (!llvm.ptr<1>) -> memref<?xf32, strided<[2]>, 1>
// CHECK:         %[[VALUE:.*]] = affine.load %[[VIEW]][%arg1] : memref<?xf32, strided<[2]>, 1>
// CHECK:         return %[[VALUE]] : f32
func.func @access_layout_mismatch(%arg0: memref<16xf32, 1>,
                                  %arg1: index) -> f32 {
  %ptr = "enzymexla.memref2pointer"(%arg0) : (memref<16xf32, 1>) -> !llvm.ptr<1>
  %view = "enzymexla.pointer2memref"(%ptr) : (!llvm.ptr<1>) -> memref<?xf32, strided<[2]>, 1>
  %value = affine.load %view[%arg1] : memref<?xf32, strided<[2]>, 1>
  return %value : f32
}
