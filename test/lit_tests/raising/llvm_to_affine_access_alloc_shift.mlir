// RUN: enzymexlamlir-opt %s --llvm-to-affine-access -split-input-file | FileCheck %s

func.func @oversized_shift(%n: index) -> i8 {
  %c64 = arith.constant 64 : index
  %size = arith.shrui %n, %c64 : index
  %alloc = memref.alloc(%size) : memref<?xf64>
  %ptr = "enzymexla.memref2pointer"(%alloc) : (memref<?xf64>) -> !llvm.ptr
  %view = "enzymexla.pointer2memref"(%ptr) : (!llvm.ptr) -> memref<?xi8>
  %c0 = arith.constant 0 : index
  %value = memref.load %view[%c0] : memref<?xi8>
  return %value : i8
}

// CHECK-LABEL: func.func @oversized_shift
// CHECK:       %[[ALLOC:.*]] = memref.alloc
// CHECK:       %[[PTR:.*]] = "enzymexla.memref2pointer"(%[[ALLOC]])
// CHECK:       "enzymexla.pointer2memref"(%[[PTR]])

// -----

func.func @valid_shift_divisibility(%n: index) -> f64 {
  %c3 = arith.constant 3 : index
  %size = arith.shli %n, %c3 : index
  %alloc = memref.alloc(%size) : memref<?xi8>
  %ptr = "enzymexla.memref2pointer"(%alloc) : (memref<?xi8>) -> !llvm.ptr
  %view = "enzymexla.pointer2memref"(%ptr) : (!llvm.ptr) -> memref<?xf64>
  %c0 = arith.constant 0 : index
  %value = memref.load %view[%c0] : memref<?xf64>
  return %value : f64
}

// CHECK-LABEL: func.func @valid_shift_divisibility
// CHECK-NOT:   enzymexla
// CHECK:       %[[ALLOC:.*]] = memref.alloc({{.*}}) : memref<?xf64>
// CHECK:       memref.load %[[ALLOC]]

// -----

func.func @overflowing_shift_scale(%n: index) -> i8 {
  %c63 = arith.constant 63 : index
  %c1 = arith.constant 1 : index
  %scaled = arith.shli %n, %c63 : index
  %size = arith.shli %scaled, %c1 : index
  %alloc = memref.alloc(%size) : memref<?xf64>
  %ptr = "enzymexla.memref2pointer"(%alloc) : (memref<?xf64>) -> !llvm.ptr
  %view = "enzymexla.pointer2memref"(%ptr) : (!llvm.ptr) -> memref<?xi8>
  %c0 = arith.constant 0 : index
  %value = memref.load %view[%c0] : memref<?xi8>
  return %value : i8
}

// CHECK-LABEL: func.func @overflowing_shift_scale
// CHECK:       %[[ALLOC:.*]] = memref.alloc
// CHECK:       %[[PTR:.*]] = "enzymexla.memref2pointer"(%[[ALLOC]])
// CHECK:       "enzymexla.pointer2memref"(%[[PTR]])

// -----

func.func @shift_derived_allocation_overflow(%n: index) -> i8 {
  %c62 = arith.constant 62 : index
  %size = arith.shli %n, %c62 : index
  %alloc = memref.alloc(%size) : memref<?xf64>
  %ptr = "enzymexla.memref2pointer"(%alloc) : (memref<?xf64>) -> !llvm.ptr
  %view = "enzymexla.pointer2memref"(%ptr) : (!llvm.ptr) -> memref<?xi8>
  %c0 = arith.constant 0 : index
  %value = memref.load %view[%c0] : memref<?xi8>
  return %value : i8
}

// CHECK-LABEL: func.func @shift_derived_allocation_overflow
// CHECK:       %[[ALLOC:.*]] = memref.alloc
// CHECK:       %[[PTR:.*]] = "enzymexla.memref2pointer"(%[[ALLOC]])
// CHECK:       "enzymexla.pointer2memref"(%[[PTR]])

// -----

func.func @constant_allocation_overflow() -> i8 {
  %size = arith.constant 4611686018427387904 : index
  %alloc = memref.alloc(%size) : memref<?xf64>
  %ptr = "enzymexla.memref2pointer"(%alloc) : (memref<?xf64>) -> !llvm.ptr
  %view = "enzymexla.pointer2memref"(%ptr) : (!llvm.ptr) -> memref<?xi8>
  %c0 = arith.constant 0 : index
  %value = memref.load %view[%c0] : memref<?xi8>
  return %value : i8
}

// CHECK-LABEL: func.func @constant_allocation_overflow
// CHECK:       %[[ALLOC:.*]] = memref.alloc
// CHECK:       %[[PTR:.*]] = "enzymexla.memref2pointer"(%[[ALLOC]])
// CHECK:       "enzymexla.pointer2memref"(%[[PTR]])

// -----

func.func @negative_allocation_extent() -> i8 {
  %size = arith.constant -1 : index
  %alloc = memref.alloc(%size) : memref<?xf64>
  %ptr = "enzymexla.memref2pointer"(%alloc) : (memref<?xf64>) -> !llvm.ptr
  %view = "enzymexla.pointer2memref"(%ptr) : (!llvm.ptr) -> memref<?xi8>
  %c0 = arith.constant 0 : index
  %value = memref.load %view[%c0] : memref<?xi8>
  return %value : i8
}

// CHECK-LABEL: func.func @negative_allocation_extent
// CHECK:       %[[ALLOC:.*]] = memref.alloc
// CHECK:       %[[PTR:.*]] = "enzymexla.memref2pointer"(%[[ALLOC]])
// CHECK:       "enzymexla.pointer2memref"(%[[PTR]])

// -----

func.func @scalable_allocation_element(%n: index) -> i8 {
  %alloc = memref.alloc(%n) : memref<?xvector<[4]xf32>>
  %ptr = "enzymexla.memref2pointer"(%alloc)
      : (memref<?xvector<[4]xf32>>) -> !llvm.ptr
  %view = "enzymexla.pointer2memref"(%ptr) : (!llvm.ptr) -> memref<?xi8>
  %c0 = arith.constant 0 : index
  %value = memref.load %view[%c0] : memref<?xi8>
  return %value : i8
}

// CHECK-LABEL: func.func @scalable_allocation_element
// CHECK:       %[[ALLOC:.*]] = memref.alloc
// CHECK:       %[[PTR:.*]] = "enzymexla.memref2pointer"(%[[ALLOC]])
// CHECK:       "enzymexla.pointer2memref"(%[[PTR]])

// -----

func.func @overflowing_right_shift_scale(%n: index) -> i8 {
  %c63 = arith.constant 63 : index
  %c1 = arith.constant 1 : index
  %scaled = arith.shrui %n, %c63 : index
  %size = arith.shrui %scaled, %c1 : index
  %alloc = memref.alloc(%size) : memref<?xf64>
  %ptr = "enzymexla.memref2pointer"(%alloc) : (memref<?xf64>) -> !llvm.ptr
  %view = "enzymexla.pointer2memref"(%ptr) : (!llvm.ptr) -> memref<?xi8>
  %c0 = arith.constant 0 : index
  %value = memref.load %view[%c0] : memref<?xi8>
  return %value : i8
}

// CHECK-LABEL: func.func @overflowing_right_shift_scale
// CHECK:       %[[ALLOC:.*]] = memref.alloc
// CHECK:       %[[PTR:.*]] = "enzymexla.memref2pointer"(%[[ALLOC]])
// CHECK:       "enzymexla.pointer2memref"(%[[PTR]])

// -----

func.func @negative_shift(%n: index) -> i8 {
  %cm1 = arith.constant -1 : index
  %size = arith.shli %n, %cm1 : index
  %alloc = memref.alloc(%size) : memref<?xf64>
  %ptr = "enzymexla.memref2pointer"(%alloc) : (memref<?xf64>) -> !llvm.ptr
  %view = "enzymexla.pointer2memref"(%ptr) : (!llvm.ptr) -> memref<?xi8>
  %c0 = arith.constant 0 : index
  %value = memref.load %view[%c0] : memref<?xi8>
  return %value : i8
}

// CHECK-LABEL: func.func @negative_shift
// CHECK:       %[[ALLOC:.*]] = memref.alloc
// CHECK:       %[[PTR:.*]] = "enzymexla.memref2pointer"(%[[ALLOC]])
// CHECK:       "enzymexla.pointer2memref"(%[[PTR]])
