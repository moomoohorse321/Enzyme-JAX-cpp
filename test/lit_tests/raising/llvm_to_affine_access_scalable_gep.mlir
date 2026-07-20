// RUN: enzymexlamlir-opt %s --llvm-to-affine-access --split-input-file | FileCheck %s

// A scalable GEP stride cannot be represented by a fixed affine constant.
// Keep the access on the exact element-indexed memref fallback instead of
// asking TypeSize to discard its scalable component.
// CHECK-LABEL: func.func @preserve_scalable_gep(
// CHECK:         %[[VIEW:.*]] = "enzymexla.pointer2memref"(%{{.*}})
// CHECK-SAME:        memref<?xvector<[4]xf32>>
// CHECK:         %[[VALUE:.*]] = memref.load %[[VIEW]][%{{.*}}]
// CHECK-NOT:     affine.load
func.func @preserve_scalable_gep(%base: !llvm.ptr, %index: i64)
    -> vector<[4]xf32> {
  %address = llvm.getelementptr %base[%index]
      : (!llvm.ptr, i64) -> !llvm.ptr, vector<[4]xf32>
  %value = llvm.load %address : !llvm.ptr -> vector<[4]xf32>
  return %value : vector<[4]xf32>
}

// -----

// Fixed-width vector strides remain eligible for the existing affine access
// recovery.
// CHECK-LABEL: func.func @raise_fixed_vector_gep(
// CHECK-NOT:     llvm.load
// CHECK:         affine.load
func.func @raise_fixed_vector_gep(%base: !llvm.ptr, %index: i64)
    -> vector<4xf32> {
  %address = llvm.getelementptr %base[%index]
      : (!llvm.ptr, i64) -> !llvm.ptr, vector<4xf32>
  %value = llvm.load %address : !llvm.ptr -> vector<4xf32>
  return %value : vector<4xf32>
}
