import unittest

import numpy as np

import bm25s


class TestEmptyQueryScores(unittest.TestCase):
    def setUp(self):
        self.retriever = bm25s.BM25(backend="numpy")
        corpus = bm25s.tokenize(
            ["a cat purrs", "a dog barks"], stopwords=None, show_progress=False
        )
        self.retriever.index(corpus, show_progress=False)

    def test_stopword_only_query_returns_zero_scores(self):
        query = bm25s.tokenize(
            "the and", stopwords="en", return_ids=False, show_progress=False
        )[0]
        self.assertEqual(query, [])
        np.testing.assert_array_equal(self.retriever.get_scores(query), np.zeros(2))
        np.testing.assert_array_equal(
            self.retriever.get_scores(query), self.retriever.get_scores_from_ids([])
        )

    def test_empty_query_respects_weight_mask(self):
        weight_mask = np.array([0.25, 1.0], dtype=np.float64)
        np.testing.assert_array_equal(
            self.retriever.get_scores([], weight_mask=weight_mask),
            self.retriever.get_scores_from_ids([], weight_mask=weight_mask),
        )

    def test_non_list_query_still_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be a list"):
            self.retriever.get_scores(())
