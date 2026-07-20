import math
from pathlib import Path
import re

from absl.testing import absltest
import jax

jax.config.update("jax_platforms", "cpu")

import jax.numpy as jnp  # noqa: E402

from enzyme_ad.jax import CPPPolygeistPipeline, cpp_call  # noqa: E402


APP_DIR = Path(__file__).parent / "cpp_polygeist_apps"
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_PATTERN.findall(text.lower())


def encode_bm25_inputs(query, documents):
    vocabulary = {}

    def encode(tokens):
        result = []
        for token in tokens:
            if token not in vocabulary:
                vocabulary[token] = len(vocabulary) + 1
            result.append(vocabulary[token])
        return result

    encoded_documents = [encode(tokenize(document)) for document in documents]
    encoded_query = encode(tokenize(query))
    width = max(map(len, encoded_documents))
    padded_documents = [
        document + [0] * (width - len(document))
        for document in encoded_documents
    ]
    return (
        jnp.asarray(padded_documents, dtype=jnp.int32),
        jnp.asarray(list(map(len, encoded_documents)), dtype=jnp.int32),
        jnp.asarray(encoded_query, dtype=jnp.int32),
    )


def reference_bm25(query, documents):
    tokenized_documents = list(map(tokenize, documents))
    query_terms = tokenize(query)
    average_length = sum(map(len, tokenized_documents)) / len(documents)
    scores = []
    for document in tokenized_documents:
        score = 0.0
        for term in query_terms:
            term_frequency = document.count(term)
            document_frequency = sum(
                term in candidate for candidate in tokenized_documents
            )
            inverse_document_frequency = math.log(
                (len(documents) - document_frequency + 0.5)
                / (document_frequency + 0.5)
                + 1.0
            )
            denominator = term_frequency + 1.2 * (
                1.0 - 0.75 + 0.75 * len(document) / average_length
            )
            score += (
                inverse_document_frequency
                * term_frequency
                * (1.2 + 1.0)
                / denominator
            )
        scores.append(score)
    return scores


def rank_bm25(query, documents):
    if not documents:
        raise ValueError("BM25 needs at least one document")
    if not tokenize(query):
        raise ValueError("BM25 needs at least one query term")

    encoded_documents, document_lengths, encoded_query = encode_bm25_inputs(
        query, documents
    )
    source = (APP_DIR / "bm25.cpp").read_text()

    @jax.jit
    def run(encoded_documents, document_lengths, encoded_query):
        (scores,) = cpp_call(
            encoded_documents,
            document_lengths,
            encoded_query,
            out_shapes=[
                jax.core.ShapedArray((len(documents),), jnp.float32)
            ],
            source=source,
            fn="bm25",
            pipeline_options=CPPPolygeistPipeline(),
        )
        return scores, jnp.argsort(-scores)

    scores, order = run(encoded_documents, document_lengths, encoded_query)
    host_scores = list(map(float, scores))
    return [
        (documents[int(index)], host_scores[int(index)]) for index in order
    ]


class CPPPolygeistTest(absltest.TestCase):
    def test_bm25(self):
        query = "quick fox search"
        documents = [
            "The quick brown fox jumps over the lazy dog",
            "Quick brown fox quick",
            "Search engines rank documents with relevance scores",
            "Fox search systems combine term frequency signals",
        ]
        ranked = rank_bm25(query, documents)
        reference_scores = reference_bm25(query, documents)
        expected_order = sorted(
            range(len(documents)),
            key=lambda index: reference_scores[index],
            reverse=True,
        )

        self.assertEqual(
            [document for document, _ in ranked],
            [documents[index] for index in expected_order],
        )
        for (_, score), index in zip(ranked, expected_order):
            self.assertAlmostEqual(score, reference_scores[index], places=5)

    def test_transpose(self):
        source = (APP_DIR / "transpose.cpp").read_text()

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
