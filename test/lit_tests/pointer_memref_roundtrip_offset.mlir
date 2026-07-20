// RUN: enzymexlamlir-opt %s --canonicalize --split-input-file | FileCheck %s

// pointer2memref records an offset in element units. memref2pointer applies
// that offset, so the round trip denotes ptr + 3 * sizeof(i32), not ptr.
// CHECK-LABEL: func.func @preserve_nonzero_offset(
// CHECK:         %[[MEMREF:.*]] = "enzymexla.pointer2memref"(%arg0)
// CHECK:         %[[POINTER:.*]] = "enzymexla.memref2pointer"(%[[MEMREF]])
// CHECK:         return %[[POINTER]]
func.func @preserve_nonzero_offset(%ptr: !llvm.ptr) -> !llvm.ptr {
  %memref = "enzymexla.pointer2memref"(%ptr)
      : (!llvm.ptr) -> memref<4xi32, strided<[1], offset: 3>>
  %roundtrip = "enzymexla.memref2pointer"(%memref)
      : (memref<4xi32, strided<[1], offset: 3>>) -> !llvm.ptr
  return %roundtrip : !llvm.ptr
}

// -----

// CHECK-LABEL: func.func @fold_zero_offset(
// CHECK-NOT:     enzymexla.pointer2memref
// CHECK-NOT:     enzymexla.memref2pointer
// CHECK:         return %arg0 : !llvm.ptr
func.func @fold_zero_offset(%ptr: !llvm.ptr) -> !llvm.ptr {
  %memref = "enzymexla.pointer2memref"(%ptr)
      : (!llvm.ptr) -> memref<4xi32, strided<[1], offset: 0>>
  %roundtrip = "enzymexla.memref2pointer"(%memref)
      : (memref<4xi32, strided<[1], offset: 0>>) -> !llvm.ptr
  return %roundtrip : !llvm.ptr
}
