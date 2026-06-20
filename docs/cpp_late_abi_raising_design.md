# C++ late-ABI raising design

## Problem

The current C++ path commits to the runtime callback ABI before the optimizer can
see the C++ function as a function.

Today `createLLVMMod` builds C++ source that contains:

```cpp
extern "C" void entry(void** __restrict__ outs, void** __restrict__ ins)
```

It then casts `outs[i]` and `ins[i]` to generated `enzyme::tensor<...>` types
and calls the user function from that wrapper. Clang lowers this generated
source to LLVM. At the JAX boundary, `_enzyme_primal_lowering` emits
`stablehlo.custom_call @jaxzyme.primal` with an integer kernel id.

That is a valid execution model for opaque C++ callees. It is not a usable shape
for C++ raising, because the raising pipeline sees the callback ABI rather than
the user function boundary.

The design goal is to keep the current user-facing `cpp_call` model, but defer
`entry(void**, void**)` generation until after C++ LLVM has had a chance to be
imported, raised, and transformed.

## Non-goals

This design does not require JAX caller operands to recover C++ callee types.
JAX may still be the final caller at runtime, but the C++ raising pipeline must
be able to run from C++ source or LLVM IR plus compiler-side metadata.

This design does not claim that arbitrary C++ object graphs, strings, virtual
dispatch, exceptions, heap-heavy code, or `std::vector` internals can always be
raised. Those cases may remain opaque or fail with diagnostics. The first target
is numeric kernel regions whose pointer/tensor boundary can be described.

This design does not add a separate inspection-only `cpp_raise` API as the main
path. A debug dump hook is useful, but the actual runtime path should use the
same staged pipeline as `cpp_call` to avoid two compilers drifting apart.

## Required invariant

The lowering to `entry(void**, void**)` is a final ABI lowering step only.

Before the final ABI step, the compiler representation must preserve a real
kernel function and a signature descriptor. The optimizer should never be forced
to rediscover the kernel boundary from a `void**` table.

## Pipeline

The new staged pipeline is:

```text
C++ source
  -> compile selected user function to LLVM without runtime entry wrapper
  -> derive or attach kernel signature metadata
  -> optionally run Enzyme LLVM AD on the real function boundary
  -> import LLVM into MLIR LLVM dialect
  -> attach signature metadata to the imported MLIR function
  -> recover pointer arguments to memrefs where metadata is available
  -> run existing raising / Polygeist transforms
  -> lower transformed kernel to the selected backend representation
  -> generate entry(void**, void**) only as the final runtime adapter
  -> register the final entry with the existing callback table
```

The current path does this instead:

```text
C++ source
  -> generate entry(void**, void**) wrapper
  -> Clang/LLVM
  -> optional Enzyme LLVM AD
  -> ORC JIT
  -> stablehlo.custom_call
```

The design changes the point at which `entry` is generated, not the fact that
the final runtime may still use the existing callback ABI.

## Signature metadata

Opaque-pointer LLVM IR does not carry pointer element types in function
signatures:

```llvm
define void @kernel(ptr %scores, ptr %term_freq, ptr %idf)
```

The raising path must not rely on typed pointer function signatures. Instead,
it needs an explicit kernel signature descriptor produced before or alongside
LLVM lowering.

The descriptor should represent each public kernel argument:

```text
index:        original argument index
name:         source/debug name when available
role:         input | output | input_output | temporary
scalar_type:  f32 | f64 | i32 | i64 | ...
rank:         integer rank
shape:        static dims or dynamic dims
layout:       initially contiguous row-major
const:        whether the source parameter is const
address_space: LLVM address space when nonzero
source_kind:  enzyme_tensor | pointer_with_contract | memref_like | unknown
```

For the existing `enzyme::tensor<T, dims...>` ABI, this descriptor can be
derived from the same `out_shapes`, `in_shapes`, `out_names`, and `in_names`
that are already passed into `createLLVMMod`.

For plain C++ pointer parameters, element type can come from Clang AST/source
type information before opaque-pointer LLVM erases it. Rank and shape are not
generally recoverable from LLVM alone; they must come from a boundary contract,
for example existing Python `out_shapes`/input avals, explicit source
annotations, or a sidecar descriptor. When the descriptor is produced from
Python shape values, it should be treated as compiler-side kernel metadata, not
as MLIR type inference from a JAX call edge.

The implementation should not depend on LLVM named metadata surviving MLIR
import. The robust route is:

1. keep the descriptor as a C++ data structure next to the LLVM module;
2. import LLVM with `translateLLVMIRToModule`;
3. locate the imported `llvm.func` for the selected kernel;
4. attach MLIR attributes derived from the descriptor directly to that function
   or its arguments before running the raise pipeline.

## Code structure

Refactor the current monolithic `createLLVMMod` into phases. The exact function
names can change, but the ownership boundaries should be clear.

### 1. Build source for the real kernel

Add a path that produces compilable C++ source without emitting
`entry(void**, void**)`.

For tensor ABI this still includes:

```cpp
#include <enzyme/tensor>
#include <enzyme/utils>
```

and the user's source. It may instantiate a selected template if needed, but it
must not wrap the kernel in the runtime callback ABI.

### 2. Compile to LLVM for raising

Add a compile mode to `GetLLVMFromJob` or a sibling function that compiles the
real kernel module for raising.

The current `GetLLVMFromJob` always runs a `default<O3>` LLVM pipeline after
Clang emits LLVM. That may be appropriate for the native callback path, but it
is the wrong default for raising if it erases structure before MLIR import. The
raising mode should make the LLVM optimization pipeline configurable and should
default to a conservative pre-raise pipeline.

### 3. Optional Enzyme LLVM AD

For C++ AD, keep using Enzyme's LLVM path. The important change is that the AD
wrapper should target the real function boundary, not the final `void**` ABI.

Forward mode can still generate a typed wrapper equivalent to:

```cpp
enzyme::__enzyme_fwddiff(fn,
                         enzyme_dup, &out, &dout,
                         enzyme_dup, &in, &din);
```

Reverse mode can still use `__enzyme_augmentfwd` and `__enzyme_reverse`.

The derivative LLVM function produced by Enzyme then becomes the function to
import and raise. Only after that path completes should the final runtime entry
be generated.

### 4. Import LLVM and attach signature metadata

Reuse the import mechanism in `raise.cpp`:

```cpp
mlir::translateLLVMIRToModule(...)
```

Immediately after import, attach argument metadata to the imported kernel
function. This avoids depending on a typed JAX caller and avoids depending on
opaque LLVM pointers to carry pointee types.

### 5. Extend pointer-to-memref recovery

`LLVMToMemrefAccess.cpp` currently recovers memref types from typed
`enzymexla.jit_call` or `enzymexla.kernel_call` operands. Add a second recovery
source:

```text
imported function argument metadata
```

When a function argument has `!llvm.ptr` and a valid kernel argument descriptor,
rewrite the function argument to a `memref<...>` matching the descriptor and
insert `enzymexla.memref2pointer` for remaining LLVM uses, following the same
bridging pattern already used by the current pass.

If no descriptor exists for a pointer argument, keep it as `!llvm.ptr` and let
later passes either preserve it or fail with a diagnostic.

### 6. Run the existing raise pipeline

After pointer recovery, run the existing pipeline from `raise.cpp`, including:

```text
llvm-to-memref-access
polygeist-mem2reg
convert-llvm-to-cf
enzyme-lift-cf-to-scf
llvm-to-affine-access
delinearize-indexing
raise-affine-to-stablehlo
```

The pipeline should be parameterized by backend and by whether the user wants a
StableHLO result or a transformed native callback result.

### 7. Generate the runtime ABI wrapper late

Only after the transform pipeline has produced the final kernel representation,
generate:

```cpp
extern "C" void entry(void** outs, void** ins)
```

or the equivalent LLVM/MLIR wrapper.

The wrapper performs the same job as today: unpack the runtime buffer table and
call the transformed kernel. The difference is that `entry` is not part of the
IR that the raising pipeline attempts to analyze.

## Public API

The main user-facing path should remain `cpp_call`.

Add an option rather than a separate primary API, for example:

```python
cpp_call(
    *args,
    out_shapes=[...],
    source=...,
    fn="bm25_score",
    pipeline_options=CPPPipeline(raise_cpp=True, ...)
)
```

The default can remain the current opaque callback path until the raised path is
complete enough to support existing users.

A debug helper that returns raised MLIR is useful, but it should call into the
same internal staged pipeline and stop before final ABI generation. It must not
be a separate compiler implementation.

## Interaction with the raw-entry ABI

The raw-entry ABI added in the fork is already a final runtime ABI:

```cpp
extern "C" void entry(void** outs, void** ins)
```

That path is useful when users explicitly want to write the runtime adapter by
hand. It is not a good raising input, because it intentionally exposes only the
buffer table ABI. The raised C++ path should start from the real user function,
not from raw entry.

## Diagnostics

The raised C++ path should provide explicit diagnostics for cases that cannot be
raised:

- pointer argument has no descriptor;
- descriptor element type conflicts with observed load/store type;
- pointer escapes through an unsupported call;
- memory layout is not contiguous row-major and no layout model is available;
- exceptions, virtual dispatch, or heap behavior prevents structured raising.

When possible, diagnostics should include the function name, argument index,
and the pass that rejected the IR.

## Tests

The design needs tests at each boundary.

### Unit tests

1. Compile a tensor-style BM25 kernel without generating `entry`.
2. Verify the LLVM module contains the selected user function or derivative
   function and does not contain the final runtime callback wrapper.
3. Import that LLVM into MLIR and verify the kernel function receives argument
   metadata attributes.
4. Run `llvm-to-memref-access` on an imported function with metadata and verify
   `!llvm.ptr` arguments become memrefs.

### Integration tests

1. `cpp_call(..., raise_cpp=True)` on a BM25-like kernel returns the same values
   as the current callback path.
2. A dump of the pre-final-wrapper MLIR contains structured memory/control-flow
   IR such as `memref`, `scf`, or `affine`.
3. The pre-final-wrapper MLIR does not contain `entry(void**, void**)`.
4. The final lowered artifact does contain exactly one runtime `entry` wrapper.

### AD tests

1. Forward-mode C++ AD for `x*x + 3*x` matches the current `cpp_call` result.
2. Reverse-mode C++ AD for the same kernel matches the current `cpp_call`
   result.
3. Dumps show Enzyme LLVM AD runs before MLIR raising and the runtime wrapper is
   still generated after raising.

## Acceptance criteria

The patch is complete when:

1. `cpp_call` can select a raised C++ pipeline without using a separate
   compiler implementation.
2. `entry(void**, void**)` is generated only after the raise/transform pipeline.
3. Opaque-pointer function args can be recovered from explicit kernel signature
   metadata without requiring a typed JAX caller.
4. The BM25 reproducer has a raised-path test that demonstrates structured MLIR
   before final ABI lowering.
5. Existing opaque callback behavior remains available for unsupported C++.
