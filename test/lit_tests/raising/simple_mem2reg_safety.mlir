// RUN: enzymexlamlir-opt %s --llvm-to-affine-access | FileCheck %s

// One static store and load execute twice. On iteration 1, iteration 0 has
// initialized storage[0] to 0, while the current store writes 1 to storage[1].
// Replacing the load with the current induction variable therefore changes the
// defined result from 0 to 1.
// CHECK-LABEL: func.func @loop_reuses_storage()
// CHECK:         %[[ZERO:[^ ]+]] = arith.constant 0 : index
// CHECK:         %[[STORAGE:[^ ]+]] = memref.alloca() : memref<2xindex>
// CHECK:         scf.for %[[I:[^ ]+]] = %[[ZERO]]
// CHECK:           memref.store %[[I]], %[[STORAGE]][%[[I]]]
// CHECK:           %[[LOADED:[^ ]+]] = memref.load %[[STORAGE]][%[[ZERO]]]
// CHECK:           scf.yield %[[LOADED]] : index
func.func @loop_reuses_storage() -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %storage = memref.alloca() : memref<2xindex>
  %result = scf.for %i = %c0 to %c2 step %c1
      iter_args(%previous = %c0) -> index {
    memref.store %i, %storage[%i] : memref<2xindex>
    %loaded = memref.load %storage[%c0] : memref<2xindex>
    scf.yield %loaded : index
  }
  return %result : index
}
