// RUN: enzymexlamlir-opt %s --llvm-to-affine-access | FileCheck %s

// A selected compatible memref is already a valid affine access base. Keep the
// select instead of creating an unnecessary scf.if from a pass that does not
// declare the SCF dialect.
// CHECK-LABEL: func.func @load_select(
// CHECK:         %[[TRUE:.*]] = "enzymexla.pointer2memref"({{.*}}) : (!llvm.ptr) -> memref<?xf32>
// CHECK:         %[[FALSE:.*]] = "enzymexla.pointer2memref"({{.*}}) : (!llvm.ptr) -> memref<?xf32>
// CHECK:         %[[SELECTED:.*]] = arith.select %{{.*}}, %[[TRUE]], %[[FALSE]] : memref<?xf32>
// CHECK:         %[[LOADED:.*]] = affine.load %[[SELECTED]][symbol(%{{.*}})] {{.*}}: memref<?xf32>
// CHECK:         return %[[LOADED]] : f32
func.func @load_select(%if_true: memref<64xf32>,
                       %if_false: memref<64xf32>, %condition: i1,
                       %index: index) -> f32 {
  %true_ptr = "enzymexla.memref2pointer"(%if_true) : (memref<64xf32>) -> !llvm.ptr
  %false_ptr = "enzymexla.memref2pointer"(%if_false) : (memref<64xf32>) -> !llvm.ptr
  %selected_ptr = arith.select %condition, %true_ptr, %false_ptr : !llvm.ptr
  %index_i64 = arith.index_castui %index : index to i64
  %address = llvm.getelementptr inbounds %selected_ptr[%index_i64]
      : (!llvm.ptr, i64) -> !llvm.ptr, f32
  %loaded = llvm.load %address : !llvm.ptr -> f32
  return %loaded : f32
}
