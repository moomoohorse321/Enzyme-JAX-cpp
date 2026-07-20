// RUN: enzymexlamlir-opt %s --llvm-to-affine-access --split-input-file | FileCheck %s

// Fixed-width add/sub wrap before the signed cast. Distributing the cast would
// instead perform the arithmetic at index width and can change the value.
// CHECK-LABEL: func.func @wrapping_add_sub(
// CHECK:         %[[ADD:.*]] = arith.addi %arg0, %arg1 : i32
// CHECK-NEXT:    %[[SUB:.*]] = arith.subi %arg0, %arg1 : i32
// CHECK-NEXT:    %[[ADD_INDEX:.*]] = arith.index_cast %[[ADD]] : i32 to index
// CHECK-NEXT:    %[[SUB_INDEX:.*]] = arith.index_cast %[[SUB]] : i32 to index
// CHECK-NEXT:    return %[[ADD_INDEX]], %[[SUB_INDEX]] : index, index
func.func @wrapping_add_sub(%a: i32, %b: i32) -> (index, index) {
  %add = arith.addi %a, %b : i32
  %sub = arith.subi %a, %b : i32
  %add_index = arith.index_cast %add : i32 to index
  %sub_index = arith.index_cast %sub : i32 to index
  return %add_index, %sub_index : index, index
}

// -----

// nsw proves that both source results fit in i16. Since i16 is no wider than
// index here, the widened arithmetic is equivalent and remains nsw.
// CHECK-LABEL: func.func @no_signed_wrap_add_sub(
// CHECK-DAG:     %[[A_ADD:.*]] = arith.index_cast %arg0 : i16 to index
// CHECK-DAG:     %[[B_ADD:.*]] = arith.index_cast %arg1 : i16 to index
// CHECK:         %[[ADD:.*]] = arith.addi %[[A_ADD]], %[[B_ADD]] overflow<nsw> : index
// CHECK-DAG:     %[[A_SUB:.*]] = arith.index_cast %arg0 : i16 to index
// CHECK-DAG:     %[[B_SUB:.*]] = arith.index_cast %arg1 : i16 to index
// CHECK:         %[[SUB:.*]] = arith.subi %[[A_SUB]], %[[B_SUB]] overflow<nsw> : index
// CHECK-NEXT:    return %[[ADD]], %[[SUB]] : index, index
func.func @no_signed_wrap_add_sub(%a: i16, %b: i16) -> (index, index) {
  %add = arith.addi %a, %b overflow<nsw> : i16
  %sub = arith.subi %a, %b overflow<nsw> : i16
  %add_index = arith.index_cast %add : i16 to index
  %sub_index = arith.index_cast %sub : i16 to index
  return %add_index, %sub_index : index, index
}

// -----

// Even nsw does not make distribution through a narrowing index cast valid.
// CHECK-LABEL: func.func @source_wider_than_index(
// CHECK:         %[[ADD:.*]] = arith.addi %arg0, %arg1 overflow<nsw> : i64
// CHECK-NEXT:    %[[CAST:.*]] = arith.index_cast %[[ADD]] : i64 to index
// CHECK-NEXT:    return %[[CAST]] : index
module attributes {
  dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>
} {
  func.func @source_wider_than_index(%a: i64, %b: i64) -> index {
    %add = arith.addi %a, %b overflow<nsw> : i64
    %cast = arith.index_cast %add : i64 to index
    return %cast : index
  }
}
