// RUN: enzymexlamlir-opt %s --pass-pipeline="builtin.module(func.func(canonicalize-loops))" | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>} {
  func.func @preserve_i64_add(%lhs: index, %rhs: index) -> i64 {
    %lhs_i64 = arith.index_castui %lhs : index to i64
    %rhs_i64 = arith.index_castui %rhs : index to i64
    %sum = arith.addi %lhs_i64, %rhs_i64 : i64
    return %sum : i64
  }
}

// CHECK-LABEL: func.func @preserve_i64_add(
// CHECK: %[[LHS:.*]] = arith.index_castui %{{.*}} : index to i64
// CHECK: %[[RHS:.*]] = arith.index_castui %{{.*}} : index to i64
// CHECK: %[[SUM:.*]] = arith.addi %[[LHS]], %[[RHS]] : i64
// CHECK: return %[[SUM]] : i64
