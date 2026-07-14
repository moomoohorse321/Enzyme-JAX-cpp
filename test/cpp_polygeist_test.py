from pathlib import Path

from absl.testing import absltest
import jax

jax.config.update("jax_platforms", "cpu")

import jax.numpy as jnp  # noqa: E402

from enzyme_ad.jax import CPPPolygeistPipeline, cpp_call  # noqa: E402


class CPPPolygeistTest(absltest.TestCase):
    def test_transpose(self):
        source = (
            Path(__file__).parent / "cpp_polygeist_apps" / "transpose.cpp"
        ).read_text()

        @jax.jit
        def run(value):
            (out,) = cpp_call(
                value,
                out_shapes=[jax.core.ShapedArray((8, 6), jnp.float32)],
                source=source,
                fn="transpose",
                pipeline_options=CPPPolygeistPipeline(),
            )
            return out

        value = jnp.arange(48, dtype=jnp.float32).reshape(6, 8)
        self.assertTrue(jnp.array_equal(run(value), value.T))


if __name__ == "__main__":
    absltest.main()
