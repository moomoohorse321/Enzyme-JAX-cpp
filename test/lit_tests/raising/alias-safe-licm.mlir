// RUN: enzymexlamlir-opt %s --pass-pipeline='builtin.module(func.func(loop-invariant-code-motion),canonicalize)' | FileCheck %s

module {
  func.func @alias_safe(%ptr: !llvm.ptr, %out: memref<4xf32>,
                        %lhs: f32, %rhs: f32) {
    %read_view = "enzymexla.pointer2memref"(%ptr)
      : (!llvm.ptr) -> memref<?xf32>
    %write_view = "enzymexla.pointer2memref"(%ptr)
      : (!llvm.ptr) -> memref<?xf32>
    affine.for %i = 0 to 4 {
      %pure = arith.addf %lhs, %rhs : f32
      %old = affine.load %read_view[0] : memref<?xf32>
      %next = arith.addf %old, %old : f32
      affine.store %next, %write_view[0] : memref<?xf32>
      affine.store %pure, %out[%i] : memref<4xf32>
    }
    return
  }
}

// CHECK-LABEL: func.func @alias_safe
// CHECK: %[[READ:.*]] = "enzymexla.pointer2memref"
// CHECK: %[[WRITE:.*]] = "enzymexla.pointer2memref"
// CHECK-NOT: affine.load
// CHECK-NOT: affine.store
// CHECK: %[[PURE:.*]] = arith.addf
// CHECK-NOT: affine.load
// CHECK-NOT: affine.store
// CHECK: affine.for %[[I:.*]] = 0 to 4 {
// CHECK-NEXT: %[[OLD:.*]] = affine.load %[[READ]][0]
// CHECK-NEXT: %[[NEXT:.*]] = arith.addf %[[OLD]], %[[OLD]]
// CHECK-NEXT: affine.store %[[NEXT]], %[[WRITE]][0]
// CHECK-NEXT: affine.store %[[PURE]], {{.*}}[%[[I]]]
