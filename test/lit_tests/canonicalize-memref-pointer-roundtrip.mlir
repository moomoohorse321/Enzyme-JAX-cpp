// RUN: enzymexlamlir-opt --canonicalize -split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @reject_static_dim0_mismatch
// CHECK: "enzymexla.pointer2memref"
func.func @reject_static_dim0_mismatch(
    %arg0: memref<4xi32>) -> memref<5xi32> {
  %ptr = "enzymexla.memref2pointer"(%arg0) : (memref<4xi32>) -> !llvm.ptr
  %view = "enzymexla.pointer2memref"(%ptr) : (!llvm.ptr) -> memref<5xi32>
  return %view : memref<5xi32>
}

// -----

// CHECK-LABEL: func.func @reject_static_offset_mismatch
// CHECK: "enzymexla.pointer2memref"
func.func @reject_static_offset_mismatch(
    %arg0: memref<4xi32, strided<[1], offset: 0>>)
    -> memref<4xi32, strided<[1], offset: 3>> {
  %ptr = "enzymexla.memref2pointer"(%arg0)
      : (memref<4xi32, strided<[1], offset: 0>>) -> !llvm.ptr
  %view = "enzymexla.pointer2memref"(%ptr)
      : (!llvm.ptr) -> memref<4xi32, strided<[1], offset: 3>>
  return %view : memref<4xi32, strided<[1], offset: 3>>
}

// -----

// CHECK-LABEL: func.func @reject_static_stride_mismatch
// CHECK: "enzymexla.pointer2memref"
func.func @reject_static_stride_mismatch(
    %arg0: memref<4xi32, strided<[2], offset: 0>>)
    -> memref<4xi32, strided<[1], offset: 0>> {
  %ptr = "enzymexla.memref2pointer"(%arg0)
      : (memref<4xi32, strided<[2], offset: 0>>) -> !llvm.ptr
  %view = "enzymexla.pointer2memref"(%ptr)
      : (!llvm.ptr) -> memref<4xi32, strided<[1], offset: 0>>
  return %view : memref<4xi32, strided<[1], offset: 0>>
}

// -----

// The pointer already denotes the source memref's logical origin, so another
// nonzero offset would apply it a second time.
// CHECK-LABEL: func.func @reject_reapplied_static_offset
// CHECK: "enzymexla.pointer2memref"
func.func @reject_reapplied_static_offset(
    %arg0: memref<4xi8, strided<[1], offset: 3>>)
    -> memref<4xi8, strided<[1], offset: 3>> {
  %ptr = "enzymexla.memref2pointer"(%arg0)
      : (memref<4xi8, strided<[1], offset: 3>>) -> !llvm.ptr
  %view = "enzymexla.pointer2memref"(%ptr)
      : (!llvm.ptr) -> memref<4xi8, strided<[1], offset: 3>>
  return %view : memref<4xi8, strided<[1], offset: 3>>
}

// -----

// CHECK-LABEL: func.func @fold_exact_zero_offset_type
// CHECK-NEXT: return %arg0 : memref<4xi32>
func.func @fold_exact_zero_offset_type(
    %arg0: memref<4xi32>) -> memref<4xi32> {
  %ptr = "enzymexla.memref2pointer"(%arg0) : (memref<4xi32>) -> !llvm.ptr
  %view = "enzymexla.pointer2memref"(%ptr) : (!llvm.ptr) -> memref<4xi32>
  return %view : memref<4xi32>
}
