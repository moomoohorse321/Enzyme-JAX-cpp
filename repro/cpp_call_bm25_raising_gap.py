"""Reproducer for the current cpp_call C++ frontend raising gap.

Run with an environment that has enzyme-ad installed, for example:

  /work/nvme/beqg/haor2/qwen-iree-venv/bin/python repro/cpp_call_bm25_raising_gap.py

The script uses the public Python-level cpp_call API on a BM25-like C++ kernel,
then dumps:

  1. the JAX/StableHLO caller IR
  2. the LLVM IR produced by Enzyme-JAX's own compile_to_llvm helper

The expected evidence is that the Python cpp_call path lowers to
stablehlo.custom_call @jaxzyme.primal, while the C++ function is compiled as a
native void** callback entry. No enzymexla.jit_call edge to a visible C++ callee
is emitted, so the LLVM-to-MLIR/Polygeist raising pipeline is not exercised by
this user-facing C++ API.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp

from enzyme_ad.jax import cpp_call
from enzyme_ad.jax import enzyme_call
from enzyme_ad.jax.primitives import LANG_CPP, resource_dir


DOCS = 4
TERMS = 5


CPP_SOURCE = r"""
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
"""


def extra_clang_args() -> tuple[str, ...]:
    """Add this cluster's GCC include path when present.

    The PyPI wheel in this environment needs this path on the aarch64 SUSE
    image. On other machines it is normally absent and skipped.
    """
    gcc_include = Path("/usr/lib64/gcc/aarch64-suse-linux/14/include")
    if gcc_include.exists():
        return ("-isystem", str(gcc_include))
    return ()


@jax.jit
def bm25_cpp_call(term_freq, idf, doc_len, query_weight):
    (scores,) = cpp_call(
        term_freq,
        idf,
        doc_len,
        query_weight,
        out_shapes=[jax.core.ShapedArray((DOCS,), jnp.float32)],
        source=CPP_SOURCE,
        fn="bm25_score",
        argv=extra_clang_args(),
    )
    return scores


def main() -> None:
    jax.config.update("jax_platforms", "cpu")

    out_dir = Path(__file__).resolve().parent / "artifacts" / "cpp_call_bm25_raising_gap"
    out_dir.mkdir(parents=True, exist_ok=True)

    term_freq = jnp.array(
        [
            [3.0, 0.0, 1.0, 0.0, 2.0],
            [0.0, 4.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 2.0, 0.0, 3.0],
            [0.0, 0.0, 0.0, 5.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    idf = jnp.array([1.7, 0.8, 1.1, 1.4, 0.6], dtype=jnp.float32)
    doc_len = jnp.array([6.0, 8.0, 10.0, 4.0], dtype=jnp.float32)
    query_weight = jnp.array([1.0, 1.0, 0.0, 1.0, 0.5], dtype=jnp.float32)

    scores = bm25_cpp_call(term_freq, idf, doc_len, query_weight)
    stablehlo_text = str(
        bm25_cpp_call.lower(term_freq, idf, doc_len, query_weight).compiler_ir(
            dialect="stablehlo"
        )
    )

    stablehlo_path = out_dir / "bm25_cpp_call_stablehlo.mlir"
    llvm_path = out_dir / "bm25_cpp_call_llvm.ll"
    stablehlo_path.write_text(stablehlo_text)

    compile_argv = ("-resource-dir", resource_dir()) + extra_clang_args()
    enzyme_call.compile_to_llvm(
        str(llvm_path),
        CPP_SOURCE,
        "bm25_score",
        [("float", [DOCS])],
        [
            ("float", [DOCS, TERMS]),
            ("float", [TERMS]),
            ("float", [DOCS]),
            ("float", [TERMS]),
        ],
        compile_argv,
        LANG_CPP,
        False,
        "",
    )
    llvm_text = llvm_path.read_text()

    assert "stablehlo.custom_call @jaxzyme.primal" in stablehlo_text
    assert "enzymexla.jit_call" not in stablehlo_text
    assert "define dso_local void @entry(ptr" in llvm_text

    print("BM25 scores:", scores)
    print("StableHLO:", stablehlo_path)
    print("LLVM IR:", llvm_path)
    print()
    print("StableHLO evidence:")
    for line in stablehlo_text.splitlines():
        if "custom_call" in line or "func.func public @main" in line:
            print(line)
    print()
    print("LLVM evidence:")
    for line in llvm_text.splitlines():
        if line.startswith("define dso_local void @entry") or "load float" in line:
            print(line)
            if "load float" in line:
                break


if __name__ == "__main__":
    main()
