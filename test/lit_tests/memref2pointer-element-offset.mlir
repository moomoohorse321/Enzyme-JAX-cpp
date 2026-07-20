// RUN: enzymexlamlir-opt %s --convert-polygeist-to-llvm='use-c-style-memref=false' | FileCheck %s

// CHECK-LABEL: llvm.func @memref2pointer_static_offset
// CHECK: %[[PTR:.+]] = llvm.getelementptr %{{.+}}[3] : (!llvm.ptr) -> !llvm.ptr, i32
// CHECK-NEXT: llvm.return %[[PTR]] : !llvm.ptr
func.func @memref2pointer_static_offset(
    %arg0: memref<4xi32, strided<[1], offset: 3>>) -> !llvm.ptr {
  %ptr = "enzymexla.memref2pointer"(%arg0)
      : (memref<4xi32, strided<[1], offset: 3>>) -> !llvm.ptr
  return %ptr : !llvm.ptr
}
