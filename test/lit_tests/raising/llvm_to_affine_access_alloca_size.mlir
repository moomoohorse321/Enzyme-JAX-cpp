// RUN: enzymexlamlir-opt %s --llvm-to-affine-access -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @raise_unsigned_i8_count()
// CHECK:         memref.alloca() : memref<255xi8>
// CHECK-NOT:     llvm.alloca
// CHECK-NOT:     enzymexla.pointer2memref
func.func @raise_unsigned_i8_count() -> i8 {
  %count = llvm.mlir.constant(-1 : i8) : i8
  %alloca = llvm.alloca %count x i8 : (i8) -> !llvm.ptr
  %view = "enzymexla.pointer2memref"(%alloca)
      : (!llvm.ptr) -> memref<255xi8>
  %c0 = arith.constant 0 : index
  %value = memref.load %view[%c0] : memref<255xi8>
  return %value : i8
}

// -----

// CHECK-LABEL: func.func @reject_allocation_size_overflow()
// CHECK:         %[[ALLOCA:.*]] = llvm.alloca %{{.*}} x i64
// CHECK:         "enzymexla.pointer2memref"(%[[ALLOCA]])
// CHECK-NOT:     memref.alloca
func.func @reject_allocation_size_overflow() -> i64 {
  %count = llvm.mlir.constant(2305843009213693952 : i64) : i64
  %alloca = llvm.alloca %count x i64 : (i64) -> !llvm.ptr
  %view = "enzymexla.pointer2memref"(%alloca)
      : (!llvm.ptr) -> memref<?xi64>
  %c0 = arith.constant 0 : index
  %value = memref.load %view[%c0] : memref<?xi64>
  return %value : i64
}

// -----

// CHECK-LABEL: func.func @reject_unrepresentable_static_extent()
// CHECK:         %[[ALLOCA:.*]] = llvm.alloca %{{.*}} x i8
// CHECK:         "enzymexla.pointer2memref"(%[[ALLOCA]])
// CHECK-NOT:     memref.alloca
func.func @reject_unrepresentable_static_extent() -> i8 {
  %count = llvm.mlir.constant(-1 : i64) : i64
  %alloca = llvm.alloca %count x i8 : (i64) -> !llvm.ptr
  %view = "enzymexla.pointer2memref"(%alloca)
      : (!llvm.ptr) -> memref<?xi8>
  %c0 = arith.constant 0 : index
  %value = memref.load %view[%c0] : memref<?xi8>
  return %value : i8
}

// -----

// CHECK-LABEL: func.func @raise_padded_i24_elements()
// CHECK:         %[[ALLOCA:.*]] = memref.alloca() : memref<8xi8>
// CHECK-NOT:     llvm.alloca
// CHECK-NOT:     enzymexla.pointer2memref
// CHECK:         memref.load %[[ALLOCA]]
func.func @raise_padded_i24_elements() -> i8 {
  %count = llvm.mlir.constant(2 : i64) : i64
  %alloca = llvm.alloca %count x i24 : (i64) -> !llvm.ptr
  %view = "enzymexla.pointer2memref"(%alloca)
      : (!llvm.ptr) -> memref<8xi8>
  %c0 = arith.constant 0 : index
  %value = memref.load %view[%c0] : memref<8xi8>
  return %value : i8
}

// -----

// CHECK-LABEL: func.func @reject_short_i24_view()
// CHECK:         %[[ALLOCA:.*]] = llvm.alloca %{{.*}} x i8
// CHECK:         "enzymexla.pointer2memref"(%[[ALLOCA]])
// CHECK-NOT:     memref.alloca
func.func @reject_short_i24_view() -> i24 {
  %count = llvm.mlir.constant(6 : i64) : i64
  %alloca = llvm.alloca %count x i8 : (i64) -> !llvm.ptr
  %view = "enzymexla.pointer2memref"(%alloca)
      : (!llvm.ptr) -> memref<2xi24>
  %c0 = arith.constant 0 : index
  %value = memref.load %view[%c0] : memref<2xi24>
  return %value : i24
}

// -----

// CHECK-LABEL: func.func @raise_exact_i24_view()
// CHECK:         %[[ALLOCA:.*]] = memref.alloca() : memref<2xi24>
// CHECK-NOT:     llvm.alloca
// CHECK-NOT:     enzymexla.pointer2memref
// CHECK:         memref.load %[[ALLOCA]]
func.func @raise_exact_i24_view() -> i24 {
  %count = llvm.mlir.constant(8 : i64) : i64
  %alloca = llvm.alloca %count x i8 : (i64) -> !llvm.ptr
  %view = "enzymexla.pointer2memref"(%alloca)
      : (!llvm.ptr) -> memref<2xi24>
  %c0 = arith.constant 0 : index
  %value = memref.load %view[%c0] : memref<2xi24>
  return %value : i24
}
