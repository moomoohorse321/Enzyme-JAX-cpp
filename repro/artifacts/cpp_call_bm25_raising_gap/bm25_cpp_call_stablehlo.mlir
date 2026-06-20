module @jit_bm25_cpp_call attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4x5xf32>, %arg1: tensor<5xf32>, %arg2: tensor<4xf32>, %arg3: tensor<5xf32>) -> (tensor<4xf32> {jax.result_info = "result"}) {
    %c = stablehlo.constant dense<1> : tensor<1xi64>
    %0 = stablehlo.custom_call @jaxzyme.primal(%c, %arg0, %arg1, %arg2, %arg3) {api_version = 3 : i32, backend_config = "backend"} : (tensor<1xi64>, tensor<4x5xf32>, tensor<5xf32>, tensor<4xf32>, tensor<5xf32>) -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
