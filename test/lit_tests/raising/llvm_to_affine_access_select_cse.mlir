// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(llvm-to-affine-access)" | FileCheck %s

module {
  func.func @select_add(%condition: i1, %a: i32, %b: i32, %d: i32,
                        %e: i32) -> i32 {
    %lhs = arith.addi %a, %b : i32
    %rhs = arith.addi %d, %e : i32
    %result = arith.select %condition, %lhs, %rhs : i32
    return %result : i32
  }
}

// CHECK-LABEL: func.func @select_add(
// CHECK-SAME:    %[[CONDITION:[[:alnum:]_]+]]: i1, %[[A:[[:alnum:]_]+]]: i32,
// CHECK-SAME:    %[[B:[[:alnum:]_]+]]: i32, %[[D:[[:alnum:]_]+]]: i32,
// CHECK-SAME:    %[[E:[[:alnum:]_]+]]: i32)
// CHECK:         %[[FIRST:.*]] = arith.select %[[CONDITION]], %[[A]], %[[D]] : i32
// CHECK:         %[[SECOND:.*]] = arith.select %[[CONDITION]], %[[B]], %[[E]] : i32
// CHECK:         %[[RESULT:.*]] = arith.addi %[[FIRST]], %[[SECOND]] : i32
// CHECK:         return %[[RESULT]] : i32
