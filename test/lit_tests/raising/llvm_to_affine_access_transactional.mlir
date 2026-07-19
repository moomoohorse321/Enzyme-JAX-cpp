// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(llvm-to-affine-access,canonicalize,func.func(affine-scalrep),canonicalize)" --split-input-file | FileCheck %s
// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(llvm-to-affine-access)" --split-input-file | FileCheck %s --check-prefix=RAW

// A low declared alignment does not prevent raising when the typed root and
// byte offset prove an element-aligned access.
// CHECK-LABEL: func.func @proven_under_aligned(
// CHECK-NOT:     llvm.load
// CHECK-NOT:     llvm.store
// CHECK-NOT:     enzymexla.pointer2memref
// CHECK:         affine.store %arg1, %arg0[symbol(%arg2)] {alignment = 1 : i64
// CHECK:         return %arg1 : i64
func.func @proven_under_aligned(%storage: memref<8xi64, 1>, %value: i64,
                                %i: index) -> i64 {
  %ptr = "enzymexla.memref2pointer"(%storage) : (memref<8xi64, 1>) -> !llvm.ptr<1>
  %i64 = arith.index_cast %i : index to i64
  %address = llvm.getelementptr inbounds %ptr[%i64] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i64
  llvm.store %value, %address {alignment = 1 : i64} : i64, !llvm.ptr<1>
  %loaded = llvm.load %address {alignment = 1 : i64} : !llvm.ptr<1> -> i64
  return %loaded : i64
}

// -----

// Alignment on the final address does not make a one-byte offset divisible.
// CHECK-LABEL: func.func @non_divisible_pair(
// CHECK-NOT:     affine.load
// CHECK-NOT:     affine.store
// CHECK-NOT:     enzymexla.pointer2memref
// CHECK:         %[[LOADED:.*]] = llvm.load %{{.*}} {alignment = 8 : i64} : !llvm.ptr<1> -> i64
// CHECK:         llvm.store %[[LOADED]], %{{.*}} {alignment = 8 : i64} : i64, !llvm.ptr<1>
// RAW-LABEL: func.func @non_divisible_pair(
// RAW-NOT:     affine.load
// RAW-NOT:     affine.store
// RAW-NOT:     enzymexla.pointer2memref
// RAW:         %[[RAW_LOADED:.*]] = llvm.load %{{.*}} {alignment = 8 : i64} : !llvm.ptr<1> -> i64
// RAW:         llvm.store %[[RAW_LOADED]], %{{.*}} {alignment = 8 : i64} : i64, !llvm.ptr<1>
func.func @non_divisible_pair(%storage: memref<8xi64, 1>) {
  %ptr = "enzymexla.memref2pointer"(%storage) : (memref<8xi64, 1>) -> !llvm.ptr<1>
  %address = llvm.getelementptr %ptr[1] : (!llvm.ptr<1>) -> !llvm.ptr<1>, i8
  %loaded = llvm.load %address {alignment = 8 : i64} : !llvm.ptr<1> -> i64
  llvm.store %loaded, %address {alignment = 8 : i64} : i64, !llvm.ptr<1>
  return
}

// -----

// One unsupported access keeps every access in the function unchanged.
// CHECK-LABEL: func.func @mixed_supported_unsupported(
// CHECK-NOT:     affine.load
// CHECK-NOT:     affine.store
// CHECK-NOT:     arith.index_cast
// CHECK:         %[[GOOD:.*]] = llvm.load %{{.*}} : !llvm.ptr<1> -> i64
// CHECK:         %[[BAD:.*]] = llvm.load %{{.*}} : !llvm.ptr<1> -> i64
// CHECK:         llvm.store %{{.*}}, %{{.*}} : i64, !llvm.ptr<1>
// RAW-LABEL: func.func @mixed_supported_unsupported(
// RAW-NOT:     affine.load
// RAW-NOT:     affine.store
// RAW-NOT:     enzymexla.pointer2memref
// RAW-NOT:     arith.index_cast
// RAW:         llvm.load
// RAW:         llvm.load
// RAW:         llvm.store
func.func @mixed_supported_unsupported(%storage: memref<8xi64, 1>, %i: i64) {
  %ptr = "enzymexla.memref2pointer"(%storage) : (memref<8xi64, 1>) -> !llvm.ptr<1>
  %good_address = llvm.getelementptr %ptr[%i] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i64
  %bad_address = llvm.getelementptr %ptr[1] : (!llvm.ptr<1>) -> !llvm.ptr<1>, i8
  %good = llvm.load %good_address : !llvm.ptr<1> -> i64
  %bad = llvm.load %bad_address : !llvm.ptr<1> -> i64
  %sum = llvm.add %good, %bad : i64
  llvm.store %sum, %good_address : i64, !llvm.ptr<1>
  return
}

// -----

// Atomic ordering cannot be represented by an ordinary affine access.
// CHECK-LABEL: func.func @atomic_access(
// CHECK-NOT:     affine.load
// CHECK-NOT:     affine.store
// CHECK:         %[[LOADED:.*]] = llvm.load %{{.*}} atomic acquire
// CHECK:         llvm.store %[[LOADED]], %{{.*}} atomic release
// RAW-LABEL: func.func @atomic_access(
// RAW-NOT:     affine.load
// RAW-NOT:     affine.store
// RAW-NOT:     enzymexla.pointer2memref
// RAW:         llvm.load %{{.*}} atomic acquire
// RAW:         llvm.store %{{.*}}, %{{.*}} atomic release
func.func @atomic_access(%storage: memref<8xi64, 1>) {
  %ptr = "enzymexla.memref2pointer"(%storage) : (memref<8xi64, 1>) -> !llvm.ptr<1>
  %loaded = llvm.load %ptr atomic acquire {alignment = 8 : i64} : !llvm.ptr<1> -> i64
  llvm.store %loaded, %ptr atomic release {alignment = 8 : i64} : i64, !llvm.ptr<1>
  return
}

// -----

// Volatile accesses must remain volatile LLVM operations.
// CHECK-LABEL: func.func @volatile_access(
// CHECK-NOT:     affine.load
// CHECK-NOT:     affine.store
// CHECK:         %[[LOADED:.*]] = llvm.load volatile %{{.*}}
// CHECK:         llvm.store volatile %[[LOADED]], %{{.*}}
// RAW-LABEL: func.func @volatile_access(
// RAW-NOT:     affine.load
// RAW-NOT:     affine.store
// RAW-NOT:     enzymexla.pointer2memref
// RAW:         llvm.load volatile
// RAW:         llvm.store volatile
func.func @volatile_access(%storage: memref<8xi64, 1>) {
  %ptr = "enzymexla.memref2pointer"(%storage) : (memref<8xi64, 1>) -> !llvm.ptr<1>
  %loaded = llvm.load volatile %ptr : !llvm.ptr<1> -> i64
  llvm.store volatile %loaded, %ptr : i64, !llvm.ptr<1>
  return
}

// -----

// A loadable LLVM-only scalar without a memref element interface keeps the
// otherwise convertible access in the same function on LLVM too.
// CHECK-LABEL: func.func @invalid_element_type(
// CHECK-NOT:     affine.load
// CHECK-NOT:     affine.store
// CHECK-NOT:     enzymexla.pointer2memref
// CHECK:         %[[VALUE:.*]] = llvm.load %{{.*}} : !llvm.ptr<1> -> !llvm.target<"spirv.Image">
// CHECK:         llvm.store %{{.*}}, %{{.*}} : i64, !llvm.ptr<1>
// CHECK:         return %[[VALUE]] : !llvm.target<"spirv.Image">
// RAW-LABEL: func.func @invalid_element_type(
// RAW-NOT:     affine.load
// RAW-NOT:     affine.store
// RAW-NOT:     enzymexla.pointer2memref
// RAW:         %[[RAW_VALUE:.*]] = llvm.load %{{.*}} : !llvm.ptr<1> -> !llvm.target<"spirv.Image">
// RAW:         llvm.store %{{.*}}, %{{.*}} : i64, !llvm.ptr<1>
// RAW:         return %[[RAW_VALUE]] : !llvm.target<"spirv.Image">
func.func @invalid_element_type(%storage: memref<8xi64, 1>, %value: i64) -> !llvm.target<"spirv.Image"> {
  %ptr = "enzymexla.memref2pointer"(%storage) : (memref<8xi64, 1>) -> !llvm.ptr<1>
  %loaded = llvm.load %ptr : !llvm.ptr<1> -> !llvm.target<"spirv.Image">
  llvm.store %value, %ptr : i64, !llvm.ptr<1>
  return %loaded : !llvm.target<"spirv.Image">
}
