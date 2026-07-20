// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(func.func(canonicalize-loops))" %s | FileCheck %s

// CHECK-LABEL: func.func @high_bit_divisor(
// CHECK:         %[[HIGH_DIVISOR:.*]] = arith.constant -56 : i8
// CHECK:         %[[HIGH_REM:.*]] = arith.remui %arg0, %[[HIGH_DIVISOR]] : i8
// CHECK-NOT:     arith.muli
// CHECK:         return %[[HIGH_REM]] : i8
func.func @high_bit_divisor(%arg0: i8) -> i8 {
  %divisor = arith.constant -56 : i8
  %negated_divisor = arith.constant 56 : i8
  %quotient = arith.divui %arg0, %divisor : i8
  %product = arith.muli %quotient, %negated_divisor : i8
  %result = arith.addi %arg0, %product : i8
  return %result : i8
}

// CHECK-LABEL: func.func @low_bit_divisor(
// CHECK:         %[[LOW_DIVISOR:.*]] = arith.constant 10 : i8
// CHECK:         %[[LOW_REM:.*]] = arith.remui %arg0, %[[LOW_DIVISOR]] : i8
// CHECK-NOT:     arith.muli
// CHECK:         return %[[LOW_REM]] : i8
func.func @low_bit_divisor(%arg0: i8) -> i8 {
  %divisor = arith.constant 10 : i8
  %negated_divisor = arith.constant -10 : i8
  %quotient = arith.divui %arg0, %divisor : i8
  %product = arith.muli %quotient, %negated_divisor : i8
  %result = arith.addi %arg0, %product : i8
  return %result : i8
}
