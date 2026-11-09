import numpy as np

from dacr.features import TfidfEmbedder, clean_prompt, embed_cached, meta_features


def test_clean_prompt_parses_list_strings():
    raw = str(["Question: 2+2?\nAnswer:", "second turn"])
    assert clean_prompt(raw) == "Question: 2+2?\nAnswer:\n\nsecond turn"
    assert "\n" in clean_prompt(raw)


def test_clean_prompt_leaves_other_text_alone():
    assert clean_prompt("plain text") == "plain text"
    assert clean_prompt("[not a list") == "[not a list"
    assert clean_prompt("['unterminated") == "['unterminated"
    assert clean_prompt(None) == ""
    assert clean_prompt(["a", "b"]) == "a\n\nb"


def test_meta_features_shape_and_signals():
    f = meta_features(["def f():\n  return 1", "What is the capital of France?"])
    assert f.shape == (2, 9)
    assert f[0, 2] > 0  # has newlines
    assert f[1, 4] == 1 and f[1, 5] == 1  # question word + question mark
    assert f[0, 6] > 0  # code keyword


def test_tfidf_embedder():
    texts = [f"topic {i % 3} words alpha beta {i}" for i in range(40)]
    emb = TfidfEmbedder(n_components=8).fit(texts).transform(texts)
    assert emb.shape == (40, 8)
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-5)


class _CountingEmbedder:
    stateless = True
    name = "counting"

    def __init__(self):
        self.calls = 0

    def fit(self, texts):
        return self

    def transform(self, texts):
        self.calls += len(texts)
        return np.array([[len(t), t.count("a")] for t in texts], dtype=np.float32)


def test_embedding_cache_reuses_and_extends(tmp_path):
    emb = _CountingEmbedder()
    first = embed_cached(emb, ["a", "bb", "a"], tmp_path)
    assert emb.calls == 2  # duplicate text embedded once
    again = embed_cached(emb, ["bb", "a"], tmp_path)
    assert emb.calls == 2
    np.testing.assert_array_equal(again, first[[1, 0]])
    embed_cached(emb, ["a", "ccc"], tmp_path)
    assert emb.calls == 3
