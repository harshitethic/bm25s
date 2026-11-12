import unittest

import numpy as np

import bm25s


class TestEmptyQueryScores(unittest.TestCase):
    def setUp(self):
        corpus = ["a cat purrs", "a dog barks"]
        self.retriever = bm25s.BM25(backend="numpy")
        self.retriever.index(
            bm25s.tokenize(corpus, show_progress=False),
            show_progress=False,
        )

    def test_empty_token_list_returns_zero_scores(self):
        scores = self.retriever.get_scores([])
        np.testing.assert_array_equal(scores, np.zeros(2, dtype=scores.dtype))
        np.testing.assert_array_equal(
            scores, self.retriever.get_scores_from_ids([])
        )

    def test_stopword_only_query_matches_retrieve(self):
        query = bm25s.tokenize(
            "the and", return_ids=False, show_progress=False
        )[0]
        self.assertEqual(query, [])
        scores = self.retriever.get_scores(query)
        retrieved = self.retriever.retrieve(
            [query], k=2, backend_selection="numpy", show_progress=False
        )
        np.testing.assert_array_equal(scores, retrieved.scores[0])

    def test_empty_query_with_weight_mask(self):
        mask = np.array([1.0, 0.0])
        scores = self.retriever.get_scores([], weight_mask=mask)
        np.testing.assert_array_equal(
            scores,
            self.retriever.get_scores_from_ids([], weight_mask=mask),
        )
        np.testing.assert_array_equal(scores, np.zeros(2, dtype=scores.dtype))

    def test_nonlist_inputs_still_raise_valueerror(self):
        with self.assertRaises(ValueError):
            self.retriever.get_scores(())
        with self.assertRaises(ValueError):
            self.retriever.get_scores(None)


if __name__ == "__main__":
    unittest.main()
