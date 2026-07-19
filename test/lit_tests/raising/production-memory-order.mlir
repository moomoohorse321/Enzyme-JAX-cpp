// RUN: FileCheck %s --check-prefix=PIPELINE --input-file=%S/../../../src/enzyme_ad/jax/raise.cpp
// RUN: enzymexlamlir-opt %s --pass-pipeline='builtin.module(canonicalize,llvm-to-tessera,tessera-apply-pdl,tessera-to-llvm)' | FileCheck %s

module {
  func.func private @unknown_effect()

  func.func @preserve_memory_order(%lhs: memref<4xi32>,
                                   %rhs: memref<4xi32>) {
    %c1 = arith.constant 1 : i32
    affine.parallel (%i) = (0) to (4) {
      affine.store %c1, %lhs[%i] : memref<4xi32>
      func.call @unknown_effect() : () -> ()
      %value = affine.load %rhs[%i] : memref<4xi32>
      affine.store %value, %lhs[%i] : memref<4xi32>
    }
    return
  }
}

// Distinct memref arguments are not necessarily disjoint, and an external
// call may affect memory. The production pipeline must preserve this order.
// PIPELINE: "canonicalize,llvm-to-tessera,tessera-apply-pdl,tessera-to-llvm,";
// CHECK-LABEL: func.func @preserve_memory_order
// CHECK: affine.parallel
// CHECK-NEXT: affine.store %{{.*}}, %{{.*}}[%{{.*}}]
// CHECK-NEXT: func.call @unknown_effect()
// CHECK-NEXT: %[[VALUE:.*]] = affine.load %{{.*}}[%{{.*}}]
// CHECK-NEXT: affine.store %[[VALUE]], %{{.*}}[%{{.*}}]
