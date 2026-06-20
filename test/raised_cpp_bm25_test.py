from absl.testing import absltest
import jax

jax.config.update("jax_platforms", "cpu")

import jax.numpy as jnp  # noqa: E402

from enzyme_ad.jax import RaisedCPPPipeline, cpp_call  # noqa: E402


class RaisedCppBM25Test(absltest.TestCase):
    def test_raised_cpp_pointer_bm25_kernel(self):
        source = r"""
        extern "C" void bm25_score(float* scores,
                                   const float* term_freq,
                                   const float* idf,
                                   const float* doc_len,
                                   const float* query_weight) {
          constexpr int Docs = 4;
          constexpr int Terms = 5;
          constexpr float k1 = 1.2f;
          constexpr float b = 0.75f;
          constexpr float avg_doc_len = 7.0f;

          for (int d = 0; d < Docs; ++d) {
            float score = 0.0f;
            float len_norm = k1 * (1.0f - b + b * doc_len[d] / avg_doc_len);
            for (int t = 0; t < Terms; ++t) {
              float tf = term_freq[d * Terms + t];
              float denom = tf + len_norm;
              float saturated_tf = (tf * (k1 + 1.0f)) / denom;
              score += idf[t] * saturated_tf * query_weight[t];
            }
            scores[d] = score;
          }
        }
        """

        @jax.jit
        def score(term_freq, idf, doc_len, query_weight):
            (scores,) = cpp_call(
                term_freq,
                idf,
                doc_len,
                query_weight,
                out_shapes=[jax.core.ShapedArray((4,), jnp.float32)],
                source=source,
                fn="bm25_score",
                pipeline_options=RaisedCPPPipeline(),
            )
            return scores

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

        k1 = jnp.float32(1.2)
        b = jnp.float32(0.75)
        avg_doc_len = jnp.float32(7.0)
        len_norm = k1 * (1.0 - b + b * doc_len[:, None] / avg_doc_len)
        saturated_tf = (term_freq * (k1 + 1.0)) / (term_freq + len_norm)
        expected = jnp.sum(idf * saturated_tf * query_weight, axis=1)

        self.assertTrue(
            jnp.allclose(score(term_freq, idf, doc_len, query_weight), expected)
        )


if __name__ == "__main__":
    absltest.main()
