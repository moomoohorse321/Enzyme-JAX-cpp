from absl.testing import absltest
import jax
import jax.numpy as jnp
from enzyme_ad.jax import RaisedCPPPipeline, cpp_call, enzyme_jax_ir
import test_utils

jax.config.update("jax_platforms", "cpu")

argv = test_utils.argv


class EnzymePipeline(absltest.TestCase):
    def test_pipeline(self):
        def fn(x):
            return x

        x = jnp.ones(3)
        if hasattr(fn, "trace"):
            module = jax.jit(fn).trace(x).lower().compiler_ir(dialect="stablehlo")
            # Only applies if jax-mlir and enzyme-mlir are built on the same version
            # optimize_module(module)
            print(str(module))


class EnzymeJax(absltest.TestCase):
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
                argv=argv,
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

        self.assertTrue(jnp.allclose(score(term_freq, idf, doc_len, query_weight), expected))

    def test_custom_cpp_kernel(self):
        @jax.jit
        def do_something(ones):
            shape = jax.core.ShapedArray(ones.shape, ones.dtype)
            a, b = cpp_call(
                ones,
                out_shapes=[shape, shape],
                source="""
        template<std::size_t N, std::size_t M>
        void myfn(enzyme::tensor<float, N, M>& out0,
                  enzyme::tensor<float, N, M>& out1,
                  const enzyme::tensor<float, N, M>& in0) {
          for (int j=0; j<N; j++) {
            for (int k=0; k<M; k++) {
                out0[j][k] = in0[j][k] + 42;
            }
          }
          for (int j=0; j<2; j++) {
            for (int k=0; k<3; k++) {
                out1[j][k] = in0[j][k] + 2 * 42;
            }
          }
        }
        """,
                fn="myfn",
                argv=argv,
            )
            c = cpp_call(
                a,
                out_shapes=[jax.core.ShapedArray([4, 4], jnp.float32)],
                source="""
        template<typename T1, typename T2>
        void f(T1& out0, const T2& in1) {
          out0 = 56.0f;
        }
        """,
                argv=argv,
            )
            return a, b, c

        ones = jnp.ones((2, 3), jnp.float32)
        x, y, z = do_something(ones)

        self.assertTrue((x == 43).all())
        self.assertTrue((y == 85).all())
        self.assertTrue((z[0] == 56).all())

        # JVP
        primals, tangents = jax.jvp(do_something, (ones,), (ones,))
        self.assertTrue((primals[0] == 43).all())
        self.assertTrue((primals[1] == 85).all())
        self.assertTrue((primals[2][0] == 56).all())
        self.assertTrue((tangents[0] == 1).all())
        self.assertTrue((tangents[1] == 1).all())
        self.assertTrue((tangents[2][0] == 0).all())

        # VJP
        primals, f_vjp = jax.vjp(do_something, ones)
        (grads,) = f_vjp((x, y, z))
        self.assertTrue((primals[0] == 43).all())
        self.assertTrue((primals[1] == 85).all())
        self.assertTrue((primals[2][0] == 56).all())

        self.assertTrue(
            (
                grads[1]
                == jnp.array(
                    [
                        [128.0, 128.0, 128.0],
                    ]
                )
            ).all()
        )

    def test_enzyme_mlir_jit(self):
        @jax.jit
        @enzyme_jax_ir(argv=argv)
        def add_one(x: jax.Array, y) -> jax.Array:
            return x + 1 + y

        add_one(jnp.array([1.0, 2.0, 3.0]), jnp.array([10.0, 20.0, 30.0]))

        primals, tangents = jax.jvp(
            add_one,
            (jnp.array([1.0, 2.0, 3.0]), jnp.array([10.0, 20.0, 30.0])),
            (jnp.array([0.1, 0.2, 0.3]), jnp.array([50.0, 70.0, 110.0])),
        )
        self.assertTrue(
            (
                primals
                == jnp.array(
                    [
                        [12.0, 23.0, 34.0],
                    ]
                )
            ).all()
        )
        self.assertTrue(
            (
                tangents
                == jnp.array(
                    [
                        [50.1, 70.2, 110.3],
                    ]
                )
            ).all()
        )

        primals, f_vjp = jax.vjp(
            add_one, jnp.array([1.0, 2.0, 3.0]), jnp.array([10.0, 20.0, 30.0])
        )
        grads = f_vjp(jnp.array([500.0, 700.0, 110.0]))
        self.assertTrue(
            (
                primals
                == jnp.array(
                    [
                        [12.0, 23.0, 34.0],
                    ]
                )
            ).all()
        )
        self.assertTrue(
            (
                grads[0]
                == jnp.array(
                    [
                        [500.0, 700.0, 110.0],
                    ]
                )
            ).all()
        )
        self.assertTrue(
            (
                grads[1]
                == jnp.array(
                    [
                        [500.0, 700.0, 110.0],
                    ]
                )
            ).all()
        )


if __name__ == "__main__":
    absltest.main()
