// RUN: enzymexlamlir-opt %s --llvm-to-affine-access -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @raise_matching_dynamic_alloca(
// CHECK-SAME:      %[[COUNT:.*]]: i64
// CHECK:         %[[EXTENT:.*]] = arith.index_castui %[[COUNT]] : i64 to index
// CHECK:         %[[ALLOCA:.*]] = memref.alloca(%[[EXTENT]]) {alignment = 64 : i64} : memref<?xf32, 5>
// CHECK-NOT:     llvm.alloca
// CHECK-NOT:     enzymexla.pointer2memref
// CHECK:         memref.copy %arg1, %[[ALLOCA]] : memref<?xf32, 5> to memref<?xf32, 5>
// CHECK:         memref.copy %[[ALLOCA]], %arg2 : memref<?xf32, 5> to memref<?xf32, 5>
func.func @raise_matching_dynamic_alloca(
    %count: i64, %source: memref<?xf32, 5>,
    %destination: memref<?xf32, 5>) {
  %alloca = llvm.alloca %count x f32 {alignment = 64 : i64}
      : (i64) -> !llvm.ptr<5>
  %view = "enzymexla.pointer2memref"(%alloca)
      : (!llvm.ptr<5>) -> memref<?xf32, 5>
  memref.copy %source, %view : memref<?xf32, 5> to memref<?xf32, 5>
  memref.copy %view, %destination : memref<?xf32, 5> to memref<?xf32, 5>
  return
}

// -----

// CHECK-LABEL: func.func @reject_element_type_change(
// CHECK:         %[[ALLOCA:.*]] = llvm.alloca %{{.*}} x f32
// CHECK:         %[[VIEW:.*]] = "enzymexla.pointer2memref"(%[[ALLOCA]]) : (!llvm.ptr) -> memref<?xi32>
// CHECK-NOT:     memref.alloca
func.func @reject_element_type_change(%count: i64, %source: memref<?xi32>) {
  %alloca = llvm.alloca %count x f32 : (i64) -> !llvm.ptr
  %view = "enzymexla.pointer2memref"(%alloca)
      : (!llvm.ptr) -> memref<?xi32>
  memref.copy %source, %view : memref<?xi32> to memref<?xi32>
  return
}

// -----

// CHECK-LABEL: func.func @reject_static_view(
// CHECK:         %[[ALLOCA:.*]] = llvm.alloca %{{.*}} x f32
// CHECK:         %[[VIEW:.*]] = "enzymexla.pointer2memref"(%[[ALLOCA]]) : (!llvm.ptr) -> memref<4xf32>
// CHECK-NOT:     memref.alloca
func.func @reject_static_view(%count: i64, %source: memref<4xf32>) {
  %alloca = llvm.alloca %count x f32 : (i64) -> !llvm.ptr
  %view = "enzymexla.pointer2memref"(%alloca)
      : (!llvm.ptr) -> memref<4xf32>
  memref.copy %source, %view : memref<4xf32> to memref<4xf32>
  return
}

// -----

// CHECK-LABEL: func.func @reject_non_identity_view(
// CHECK:         %[[ALLOCA:.*]] = llvm.alloca %{{.*}} x f32
// CHECK:         %[[VIEW:.*]] = "enzymexla.pointer2memref"(%[[ALLOCA]])
// CHECK-SAME:        memref<?xf32, strided<[2]>>
// CHECK-NOT:     memref.alloca
func.func @reject_non_identity_view(
    %count: i64, %source: memref<?xf32, strided<[2], offset: 0>>) {
  %alloca = llvm.alloca %count x f32 : (i64) -> !llvm.ptr
  %view = "enzymexla.pointer2memref"(%alloca)
      : (!llvm.ptr) -> memref<?xf32, strided<[2], offset: 0>>
  memref.copy %source, %view
      : memref<?xf32, strided<[2], offset: 0>> to memref<?xf32, strided<[2], offset: 0>>
  return
}

// -----

// CHECK-LABEL: func.func @reject_address_space_change(
// CHECK:         %[[ALLOCA:.*]] = llvm.alloca %{{.*}} x f32 : (i64) -> !llvm.ptr<5>
// CHECK:         %[[VIEW:.*]] = "enzymexla.pointer2memref"(%[[ALLOCA]]) : (!llvm.ptr<5>) -> memref<?xf32, 3>
// CHECK-NOT:     memref.alloca
func.func @reject_address_space_change(%count: i64,
                                       %source: memref<?xf32, 3>) {
  %alloca = llvm.alloca %count x f32 : (i64) -> !llvm.ptr<5>
  %view = "enzymexla.pointer2memref"(%alloca)
      : (!llvm.ptr<5>) -> memref<?xf32, 3>
  memref.copy %source, %view : memref<?xf32, 3> to memref<?xf32, 3>
  return
}

// -----

// CHECK-LABEL: func.func @reject_invalid_alignment(
// CHECK:         %[[ALLOCA:.*]] = llvm.alloca %{{.*}} x f32 {alignment = 3 : i64}
// CHECK:         %[[VIEW:.*]] = "enzymexla.pointer2memref"(%[[ALLOCA]]) : (!llvm.ptr) -> memref<?xf32>
// CHECK-NOT:     memref.alloca
func.func @reject_invalid_alignment(%count: i64, %source: memref<?xf32>) {
  %alloca = llvm.alloca %count x f32 {alignment = 3 : i64}
      : (i64) -> !llvm.ptr
  %view = "enzymexla.pointer2memref"(%alloca)
      : (!llvm.ptr) -> memref<?xf32>
  memref.copy %source, %view : memref<?xf32> to memref<?xf32>
  return
}

// -----

// CHECK-LABEL: func.func @reject_inalloca(
// CHECK:         %[[ALLOCA:.*]] = llvm.alloca inalloca %{{.*}} x f32
// CHECK:         %[[VIEW:.*]] = "enzymexla.pointer2memref"(%[[ALLOCA]]) : (!llvm.ptr) -> memref<?xf32>
// CHECK-NOT:     memref.alloca
func.func @reject_inalloca(%count: i64, %source: memref<?xf32>) {
  %alloca = llvm.alloca inalloca %count x f32 : (i64) -> !llvm.ptr
  %view = "enzymexla.pointer2memref"(%alloca)
      : (!llvm.ptr) -> memref<?xf32>
  memref.copy %source, %view : memref<?xf32> to memref<?xf32>
  return
}
