Title: Investigate exposing a pure C++ LLVM-to-MLIR raising path in the opaque-pointer era

## Summary

Enzyme-JAX currently supports C++ through the Python `cpp_call` API, and the
kernel can run correctly. However, for a user who wants the C++ frontend to use
Polygeist-style raising and transforms, the current exposed path does not appear
to provide that.

The current user-visible C++ path is effectively:

```text
C++ source
  -> Clang/LLVM
  -> optional Enzyme LLVM AD
  -> native callback/JIT entry
  -> called from JAX as an opaque custom call
```

That is a valid execution and AD model for opaque C++ callees. The missing piece
is different: there does not seem to be a public path for:

```text
C++ source
  -> LLVM IR with opaque pointers
  -> MLIR LLVM dialect
  -> recover enough memory/type structure
  -> Polygeist/raising transforms
  -> structured MLIR / StableHLO / lowered executable
```

This matters because modern LLVM IR uses opaque pointers. After Clang lowers
C++ to LLVM, function signatures and calls use `ptr`, not typed pointer
signatures such as `float*` or `tensor<float, N>*`. A raising pipeline that was
designed around pre-opaque-pointer LLVM cannot rely on pointer element types
being present in function types.

The issue to investigate is not whether JAX can provide typed operands for a
callback. The target use case is a pure C++ compilation pipeline: compile C++
ahead of time or just-in-time, then reuse the repo's LLVM-to-MLIR/Polygeist
raising infrastructure on that C++ artifact without requiring a JAX caller to
be the source of type information.

The corresponding design proposal is in:

```text
docs/cpp_late_abi_raising_design.md
```

## Reproducer

I used a BM25-like numeric C++ kernel through the public Python `cpp_call` API.
The repro script is:

```bash
python repro/cpp_call_bm25_raising_gap.py
```

The script:

1. Calls a C++ BM25 kernel through `enzyme_ad.jax.cpp_call`.
2. Dumps the lowered StableHLO caller IR.
3. Dumps the LLVM IR produced by Enzyme-JAX's own `enzyme_call.compile_to_llvm`
   helper for the same C++ source.

The C++ function is intentionally accepted by today's API:

```cpp
template<enzyme::size_t Docs, enzyme::size_t Terms>
void bm25_score(enzyme::tensor<float, Docs>& scores,
                const enzyme::tensor<float, Docs, Terms>& term_freq,
                const enzyme::tensor<float, Terms>& idf,
                const enzyme::tensor<float, Docs>& doc_len,
                const enzyme::tensor<float, Terms>& query_weight) {
  constexpr float k1 = 1.2f;
  constexpr float b = 0.75f;
  constexpr float avg_doc_len = 7.0f;

  for (enzyme::size_t d = 0; d < Docs; ++d) {
    float score = 0.0f;
    float len_norm = k1 * (1.0f - b + b * doc_len[d] / avg_doc_len);
    for (enzyme::size_t t = 0; t < Terms; ++t) {
      float tf = term_freq[d][t];
      float denom = tf + len_norm;
      float saturated_tf = (tf * (k1 + 1.0f)) / denom;
      score += idf[t] * saturated_tf * query_weight[t];
    }
    scores[d] = score;
  }
}
```

## Observed behavior

The kernel runs successfully:

```text
BM25 scores: [3.185557  2.643879  2.5588477 3.012428 ]
```

The lowered StableHLO is an opaque custom call:

```mlir
module @jit_bm25_cpp_call {
  func.func public @main(
      %arg0: tensor<4x5xf32>,
      %arg1: tensor<5xf32>,
      %arg2: tensor<4xf32>,
      %arg3: tensor<5xf32>) -> tensor<4xf32> {
    %c = stablehlo.constant dense<1> : tensor<1xi64>
    %0 = stablehlo.custom_call @jaxzyme.primal(%c, %arg0, %arg1, %arg2, %arg3)
      : (tensor<1xi64>, tensor<4x5xf32>, tensor<5xf32>, tensor<4xf32>, tensor<5xf32>)
        -> tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}
```

The dumped LLVM IR for the C++ side has an opaque callback ABI:

```llvm
define dso_local void @entry(ptr %outs, ptr %ins) {
entry:
  %0 = load ptr, ptr %outs
  %1 = load ptr, ptr %ins
  ...
  %5 = load float, ptr %arrayidx.i.i
  ...
  store float %add15.i, ptr %arrayidx.i34.i
}
```

This is not evidence that the kernel is wrong. It is evidence that the current
Python C++ API executes the kernel as an opaque callback, not as a C++ body
raised into structured MLIR.

The full generated artifacts are:

```text
repro/artifacts/cpp_call_bm25_raising_gap/bm25_cpp_call_stablehlo.mlir
repro/artifacts/cpp_call_bm25_raising_gap/bm25_cpp_call_llvm.ll
```

## Core technical problem

In opaque-pointer LLVM IR, pointer-typed function arguments and call operands no
longer carry pointee types:

```llvm
define void @kernel(ptr %scores, ptr %term_freq, ptr %idf, ptr %doc_len)
```

Even if the C++ source has a C++ caller and a C++ callee, after lowering both
sides still use `ptr` at the ABI level. The useful type information is no
longer in the pointer type itself. It may exist only in places such as:

- scalar operation types, for example `load float` and `store float`;
- `getelementptr` source element types;
- debug metadata, if emitted and preserved;
- frontend/source information that could be exported as side metadata;
- user-provided or compiler-generated ABI/type metadata.

Therefore, the right question is:

> What is the correct non-hacky way for Enzyme-JAX to raise C++ LLVM IR in the
> opaque-pointer era?

This should be answerable without using JAX as a source of type information.
The C++ compiler pipeline itself should preserve, infer, or explicitly attach
the information needed by the raising passes.

## Existing infrastructure that looks relevant

The repo already contains most of the pieces:

- `src/enzyme_ad/jax/clang_compile.cc` compiles C++ to LLVM.
- `src/enzyme_ad/jax/raise.cpp` imports LLVM IR into MLIR and defines a raising
  pipeline containing passes such as `llvm-to-memref-access`,
  `polygeist-mem2reg`, `convert-llvm-to-cf`, `llvm-to-affine-access`,
  `delinearize-indexing`, and `raise-affine-to-stablehlo`.
- `src/enzyme_ad/jax/Passes/LLVMToMemrefAccess.cpp` contains the current
  pointer-to-memref recovery logic.
- `enzymexlamlir-opt` exposes the MLIR pass infrastructure for hand-written or
  already-produced MLIR.

What appears missing is a user-facing pure C++ path that connects:

```text
clang_compile.cc C++ -> LLVM
```

to:

```text
raise.cpp LLVM -> MLIR -> Polygeist/raising transforms
```

without routing through the opaque `stablehlo.custom_call @jaxzyme.*` callback
path.

## Request

Please consider adding or documenting a pure C++ raising path that:

1. accepts ordinary C++ source or LLVM IR;
2. compiles/imports it in opaque-pointer LLVM mode;
3. reconstructs enough memory structure for pointer arguments and memory ops;
4. runs the existing Polygeist/raising transforms;
5. returns or emits raised MLIR;
6. can later be connected to Python/JAX, but does not rely on JAX for type
   recovery.

If this is currently unsupported, the README/API docs should clarify that the
Python `cpp_call` path uses Enzyme LLVM AD plus native callback execution, while
the Polygeist/raising pipeline is not currently exposed as the C++ frontend path.

## Acceptance test idea

A good regression test would compile a pure C++ BM25-like numeric kernel through
the new path and check that the output contains structured memory/control-flow
IR, for example memref/affine/scf or StableHLO, rather than only:

```mlir
stablehlo.custom_call @jaxzyme.primal(...)
```

or:

```llvm
define void @entry(ptr %outs, ptr %ins)
```
