// RUN: enzymexlamlir-opt --canonicalize -split-input-file %s | FileCheck %s

// memref.cast preserves its source descriptor; pointer2memref rebuilds the
// descriptor from its result type, so the cast must remain outside the bridge.
// CHECK-LABEL: func.func @preserve_static_shape_descriptor
// CHECK: %[[MEMREF:.+]] = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<4xi32>
// CHECK: %[[CAST:.+]] = memref.cast %[[MEMREF]] : memref<4xi32> to memref<?xi32>
// CHECK: return %[[CAST]]
func.func @preserve_static_shape_descriptor(
    %arg0: !llvm.ptr) -> memref<?xi32> {
  %memref = "enzymexla.pointer2memref"(%arg0)
      : (!llvm.ptr) -> memref<4xi32>
  %cast = memref.cast %memref : memref<4xi32> to memref<?xi32>
  return %cast : memref<?xi32>
}

// -----

// CHECK-LABEL: func.func @preserve_static_offset_descriptor
// CHECK: %[[MEMREF:.+]] = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<4xi32, strided<[1]>>
// CHECK: %[[CAST:.+]] = memref.cast %[[MEMREF]] : memref<4xi32, strided<[1]>> to memref<4xi32, strided<[1], offset: ?>>
// CHECK: return %[[CAST]]
func.func @preserve_static_offset_descriptor(
    %arg0: !llvm.ptr) -> memref<4xi32, strided<[1], offset: ?>> {
  %memref = "enzymexla.pointer2memref"(%arg0)
      : (!llvm.ptr) -> memref<4xi32, strided<[1]>>
  %cast = memref.cast %memref
      : memref<4xi32, strided<[1]>>
        to memref<4xi32, strided<[1], offset: ?>>
  return %cast : memref<4xi32, strided<[1], offset: ?>>
}

// -----

// CHECK-LABEL: func.func @preserve_static_stride_descriptor
// CHECK: %[[MEMREF:.+]] = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<4xi32, strided<[1]>>
// CHECK: %[[CAST:.+]] = memref.cast %[[MEMREF]] : memref<4xi32, strided<[1]>> to memref<4xi32, strided<[?]>>
// CHECK: return %[[CAST]]
func.func @preserve_static_stride_descriptor(
    %arg0: !llvm.ptr) -> memref<4xi32, strided<[?]>> {
  %memref = "enzymexla.pointer2memref"(%arg0)
      : (!llvm.ptr) -> memref<4xi32, strided<[1]>>
  %cast = memref.cast %memref
      : memref<4xi32, strided<[1]>> to memref<4xi32, strided<[?]>>
  return %cast : memref<4xi32, strided<[?]>>
}

// -----

// CHECK-LABEL: func.func @fold_redundant_exact_type_cast
// CHECK: %[[MEMREF:.+]] = "enzymexla.pointer2memref"(%arg0) : (!llvm.ptr) -> memref<4xi32>
// CHECK-NEXT: return %[[MEMREF]]
func.func @fold_redundant_exact_type_cast(
    %arg0: !llvm.ptr) -> memref<4xi32> {
  %memref = "enzymexla.pointer2memref"(%arg0)
      : (!llvm.ptr) -> memref<4xi32>
  %cast = memref.cast %memref : memref<4xi32> to memref<4xi32>
  return %cast : memref<4xi32>
}
