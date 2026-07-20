// RUN: enzymexlamlir-opt %s --affine-cfg | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>} {
  func.func @lossy_integer_roundtrips(%wide: i64, %idx: index) -> (i64, index, i64, index) {
    %sidx = arith.index_cast %wide : i64 to index
    %swide = arith.index_cast %sidx : index to i64
    %snarrow = arith.index_cast %idx : index to i16
    %sidx_again = arith.index_cast %snarrow : i16 to index
    %uidx = arith.index_castui %wide : i64 to index
    %uwide = arith.index_castui %uidx : index to i64
    %unarrow = arith.index_castui %idx : index to i16
    %uidx_again = arith.index_castui %unarrow : i16 to index
    return %swide, %sidx_again, %uwide, %uidx_again : i64, index, i64, index
  }

  // CHECK-LABEL: func.func @lossy_integer_roundtrips
  // CHECK: %[[STRUNC:.*]] = arith.trunci %{{.*}} : i64 to i32
  // CHECK: %[[SWIDE:.*]] = arith.extsi %[[STRUNC]] : i32 to i64
  // CHECK: %[[SIDX:.*]] = arith.index_cast %{{.*}} : index to i32
  // CHECK: %[[SNARROW:.*]] = arith.trunci %[[SIDX]] : i32 to i16
  // CHECK: %[[SEXT:.*]] = arith.extsi %[[SNARROW]] : i16 to i32
  // CHECK: %[[SIDX_AGAIN:.*]] = arith.index_cast %[[SEXT]] : i32 to index
  // CHECK: %[[UTRUNC:.*]] = arith.trunci %{{.*}} : i64 to i32
  // CHECK: %[[UWIDE:.*]] = arith.extui %[[UTRUNC]] : i32 to i64
  // CHECK: %[[UIDX:.*]] = arith.index_castui %{{.*}} : index to i32
  // CHECK: %[[UNARROW:.*]] = arith.trunci %[[UIDX]] : i32 to i16
  // CHECK: %[[UEXT:.*]] = arith.extui %[[UNARROW]] : i16 to i32
  // CHECK: %[[UIDX_AGAIN:.*]] = arith.index_castui %[[UEXT]] : i32 to index
  // CHECK: return %[[SWIDE]], %[[SIDX_AGAIN]], %[[UWIDE]], %[[UIDX_AGAIN]]

  func.func @lossless_integer_roundtrips(%narrow: i16, %idx: index) -> (i16, index, i16, index) {
    %sidx = arith.index_cast %narrow : i16 to index
    %snarrow = arith.index_cast %sidx : index to i16
    %swide = arith.index_cast %idx : index to i64
    %sidx_again = arith.index_cast %swide : i64 to index
    %uidx = arith.index_castui %narrow : i16 to index
    %unarrow = arith.index_castui %uidx : index to i16
    %uwide = arith.index_castui %idx : index to i64
    %uidx_again = arith.index_castui %uwide : i64 to index
    return %snarrow, %sidx_again, %unarrow, %uidx_again : i16, index, i16, index
  }

  // CHECK-LABEL: func.func @lossless_integer_roundtrips
  // CHECK-NEXT: return %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : i16, index, i16, index

  func.func @signed_constants() -> (index, index) {
    %small = arith.constant -7 : i64
    %large = arith.constant 4294967296 : i64
    %small_idx = arith.index_cast %small : i64 to index
    %large_idx = arith.index_cast %large : i64 to index
    return %small_idx, %large_idx : index, index
  }

  // CHECK-LABEL: func.func @signed_constants
  // CHECK: %[[SIGNED_SMALL_IDX:.*]] = arith.constant -7 : index
  // CHECK: %[[SIGNED_LARGE_IDX:.*]] = arith.constant 0 : index
  // CHECK: return %[[SIGNED_SMALL_IDX]], %[[SIGNED_LARGE_IDX]]

  func.func @unsigned_constants() -> (index, index, index) {
    %small = arith.constant 7 : i16
    %negative = arith.constant -1 : i16
    %large = arith.constant 4294967296 : i64
    %small_idx = arith.index_castui %small : i16 to index
    %negative_idx = arith.index_castui %negative : i16 to index
    %large_idx = arith.index_castui %large : i64 to index
    return %small_idx, %negative_idx, %large_idx : index, index, index
  }

  // CHECK-LABEL: func.func @unsigned_constants
  // CHECK: %[[UNSIGNED_SMALL_IDX:.*]] = arith.constant 7 : index
  // CHECK: %[[UNSIGNED_NEGATIVE_IDX:.*]] = arith.constant 65535 : index
  // CHECK: %[[UNSIGNED_LARGE_IDX:.*]] = arith.constant 0 : index
  // CHECK: return %[[UNSIGNED_SMALL_IDX]], %[[UNSIGNED_NEGATIVE_IDX]], %[[UNSIGNED_LARGE_IDX]]
}
