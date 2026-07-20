// RUN: enzymexlamlir-opt --canonicalize -split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @preserve_addrspacecast(
// CHECK-SAME:    %[[ARG:.*]]: !llvm.ptr<1>
func.func @preserve_addrspacecast(%arg: !llvm.ptr<1>) -> memref<?xi8, 3> {
  // CHECK: %[[CAST:.*]] = llvm.addrspacecast %[[ARG]] : !llvm.ptr<1> to !llvm.ptr
  %cast = llvm.addrspacecast %arg : !llvm.ptr<1> to !llvm.ptr

  // CHECK: %[[VIEW:.*]] = "enzymexla.pointer2memref"(%[[CAST]]) : (!llvm.ptr) -> memref<?xi8, 3>
  %view = "enzymexla.pointer2memref"(%cast) : (!llvm.ptr) -> memref<?xi8, 3>
  return %view : memref<?xi8, 3>
}

// -----

// CHECK-LABEL: func.func @fold_bitcast(
// CHECK-SAME:    %[[ARG:.*]]: !llvm.ptr
func.func @fold_bitcast(%arg: !llvm.ptr) -> memref<?xi8> {
  // CHECK-NOT: llvm.bitcast
  %cast = llvm.bitcast %arg : !llvm.ptr to !llvm.ptr

  // CHECK: %[[VIEW:.*]] = "enzymexla.pointer2memref"(%[[ARG]]) : (!llvm.ptr) -> memref<?xi8>
  %view = "enzymexla.pointer2memref"(%cast) : (!llvm.ptr) -> memref<?xi8>
  return %view : memref<?xi8>
}

// -----

// CHECK-LABEL: func.func @fold_zero_gep(
// CHECK-SAME:    %[[ARG:.*]]: !llvm.ptr
func.func @fold_zero_gep(%arg: !llvm.ptr) -> memref<?xi8> {
  // CHECK-NOT: llvm.getelementptr
  %gep = llvm.getelementptr %arg[0] : (!llvm.ptr) -> !llvm.ptr, i8

  // CHECK: %[[VIEW:.*]] = "enzymexla.pointer2memref"(%[[ARG]]) : (!llvm.ptr) -> memref<?xi8>
  %view = "enzymexla.pointer2memref"(%gep) : (!llvm.ptr) -> memref<?xi8>
  return %view : memref<?xi8>
}
