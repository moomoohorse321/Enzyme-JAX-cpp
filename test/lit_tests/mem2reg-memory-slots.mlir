// RUN: enzymexlamlir-opt %s --sroa --mem2reg | FileCheck %s

// CHECK-LABEL: func.func @promote_static_elements(
// CHECK-NOT: memref.alloca
// CHECK-NOT: memref.store
// CHECK-NOT: memref.load
// CHECK: return %arg0 : i32
func.func @promote_static_elements(%x: i32, %y: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %slot = memref.alloca() : memref<2xi32>
  memref.store %x, %slot[%c0] : memref<2xi32>
  memref.store %y, %slot[%c1] : memref<2xi32>
  %value = memref.load %slot[%c0] : memref<2xi32>
  return %value : i32
}
