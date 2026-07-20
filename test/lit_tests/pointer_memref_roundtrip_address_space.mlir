// RUN: enzymexlamlir-opt %s --canonicalize --split-input-file | FileCheck %s

// A pointer result in the memref's address space needs addrspacecast semantics;
// an LLVM bitcast across address spaces is invalid.
// CHECK-LABEL: func.func @preserve_cross_space_result(
// CHECK:         %[[MEMREF:.*]] = "enzymexla.pointer2memref"(%arg0)
// CHECK:         %[[POINTER:.*]] = "enzymexla.memref2pointer"(%[[MEMREF]])
// CHECK-NOT:     llvm.bitcast
// CHECK:         return %[[POINTER]]
func.func @preserve_cross_space_result(%ptr: !llvm.ptr<1>) -> !llvm.ptr<3> {
  %memref = "enzymexla.pointer2memref"(%ptr)
      : (!llvm.ptr<1>) -> memref<4xi32, 3>
  %roundtrip = "enzymexla.memref2pointer"(%memref)
      : (memref<4xi32, 3>) -> !llvm.ptr<3>
  return %roundtrip : !llvm.ptr<3>
}

// -----

// LLVM address-space cast chains whose source and result pointer types match
// remain foldable, even when the intermediate memref uses another space.
// CHECK-LABEL: func.func @fold_same_result_pointer(
// CHECK-NOT:     enzymexla.pointer2memref
// CHECK-NOT:     enzymexla.memref2pointer
// CHECK:         return %arg0 : !llvm.ptr<1>
func.func @fold_same_result_pointer(%ptr: !llvm.ptr<1>) -> !llvm.ptr<1> {
  %memref = "enzymexla.pointer2memref"(%ptr)
      : (!llvm.ptr<1>) -> memref<4xi32, 3>
  %roundtrip = "enzymexla.memref2pointer"(%memref)
      : (memref<4xi32, 3>) -> !llvm.ptr<1>
  return %roundtrip : !llvm.ptr<1>
}
