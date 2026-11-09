"""Query featurization: prompt cleanup, text embeddings (cached), and hand-made meta features."""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path
from typing import Protocol

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler

DEFAULT_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"


def clean_prompt(raw: object) -> str:
    """Return the prompt as plain text.

    RouterBench stores most prompts as the ``str()`` of a Python list of turns, e.g.
    ``"['Solve ...\\nQuestion: ...']"``. That hides real newlines and adds list syntax, so we
    parse such strings (safely, with ``ast.literal_eval``) and join the turns.
    """
    if raw is None:
        return ""
    if isinstance(raw, (list, tuple)):
        return "\n\n".join(str(t) for t in raw)
    s = str(raw)
    t = s.strip()
    if len(t) >= 4 and t[0] == "[" and t[-1] == "]" and t[1] in "'\"":
        try:
            val = ast.literal_eval(t)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            return s
        if isinstance(val, (list, tuple)) and all(isinstance(v, str) for v in val):
            return "\n\n".join(val)
    return s


_QWORD = re.compile(r"\b(?:how|why|what|when|where|which|who)\b", re.I)
_CODE = re.compile(r"\b(?:def|class|function|implement|python|java|javascript|sql|code|program)\b|c\+\+", re.I)
_MATH = re.compile(r"\b(?:proof|prove|theorem|lemma|derive|equation|integral|solve|calculate)\b", re.I)

META_FEATURE_NAMES = [
    "log_chars",
    "log_words",
    "log_lines",
    "has_code_block",
    "has_question_word",
    "has_question_mark",
    "code_keywords",
    "math_keywords",
    "digit_frac",
]


def meta_features(texts: list[str]) -> np.ndarray:
    out = np.zeros((len(texts), len(META_FEATURE_NAMES)), dtype=np.float32)
    for i, s in enumerate(texts):
        n = max(len(s), 1)
        out[i] = [
            np.log1p(len(s)),
            np.log1p(len(s.split())),
            np.log1p(s.count("\n")),
            float("```" in s),
            float(bool(_QWORD.search(s))),
            float("?" in s),
            np.log1p(len(_CODE.findall(s))),
            np.log1p(len(_MATH.findall(s))),
            sum(ch.isdigit() for ch in s) / n,
        ]
    return out


# --------------------------------------------------------------------------------------
# Embedders
# --------------------------------------------------------------------------------------


class Embedder(Protocol):
    name: str
    stateless: bool

    def fit(self, texts: list[str]) -> Embedder: ...

    def transform(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Wraps a sentence-transformers model. Stateless, so embeddings can be cached on disk."""

    stateless = True

    def __init__(self, model_name: str = DEFAULT_EMBEDDER, batch_size: int = 64, device: str | None = None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = None

    @property
    def name(self) -> str:
        return f"st-{self.model_name}"

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:  # pragma: no cover - optional dependency
                raise ImportError(
                    "sentence-transformers is not installed. Install the embed extra "
                    "(pip install 'dacr-router[embed]') or use --embedder tfidf."
                ) from e
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def fit(self, texts: list[str]) -> SentenceTransformerEmbedder:
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        emb = self._load().encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 1000,
            normalize_embeddings=True,
        )
        return np.asarray(emb, dtype=np.float32)

    def __getstate__(self):  # never pickle the torch model; reload lazily instead
        state = self.__dict__.copy()
        state["_model"] = None
        return state


class TfidfEmbedder:
    """TF-IDF + truncated SVD (LSA). Fast, dependency-light, fitted on training prompts only."""

    stateless = False

    def __init__(self, n_components: int = 128, max_features: int = 50_000, seed: int = 0):
        self.n_components = n_components
        self.max_features = max_features
        self.seed = seed
        self._pipe = None

    @property
    def name(self) -> str:
        return f"tfidf-svd{self.n_components}"

    def fit(self, texts: list[str]) -> TfidfEmbedder:
        vec = TfidfVectorizer(max_features=self.max_features, ngram_range=(1, 2), sublinear_tf=True)
        n_terms = len(vec.fit(texts).vocabulary_)
        k = max(2, min(self.n_components, n_terms - 1, len(texts) - 1))
        self._pipe = make_pipeline(vec, TruncatedSVD(n_components=k, random_state=self.seed), Normalizer())
        self._pipe.fit(texts)
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        if self._pipe is None:
            raise RuntimeError("TfidfEmbedder.fit must be called first")
        return self._pipe.transform(list(texts)).astype(np.float32)


def make_embedder(spec: str) -> Embedder:
    """``tfidf`` / ``tfidf:256`` or any sentence-transformers model name."""
    if spec == "tfidf" or spec.startswith("tfidf:"):
        n = int(spec.split(":", 1)[1]) if ":" in spec else 128
        return TfidfEmbedder(n_components=n)
    return SentenceTransformerEmbedder(spec)


def _text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()


def embed_cached(embedder: Embedder, texts: list[str], cache_dir: str | Path | None) -> np.ndarray:
    """Embed ``texts``; for stateless embedders reuse/extend an on-disk cache keyed by text hash."""
    if cache_dir is None or not embedder.stateless:
        return embedder.transform(texts)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / (re.sub(r"[^A-Za-z0-9_.-]+", "_", embedder.name) + ".npz")
    keys = [_text_key(t) for t in texts]
    known: dict[str, np.ndarray] = {}
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            known = dict(zip(z["keys"].tolist(), z["emb"], strict=True))
    missing = sorted({k for k in keys if k not in known})
    if missing:
        first = {k: i for i, k in reversed(list(enumerate(keys)))}
        new = embedder.transform([texts[first[k]] for k in missing])
        known.update(zip(missing, new, strict=True))
        all_keys = list(known)
        np.savez(path, keys=np.array(all_keys), emb=np.stack([known[k] for k in all_keys]))
        print(f"[features] embedded {len(missing)} new prompts (cache: {path})")
    return np.stack([known[k] for k in keys]).astype(np.float32)


class FeatureBuilder:
    """Prompt -> feature vector: ``[embedding, standardized meta features]``."""

    def __init__(self, embedder: Embedder, cache_dir: str | Path | None = None):
        self.embedder = embedder
        self.cache_dir = cache_dir
        self.scaler = StandardScaler()

    def fit(self, texts: list[str]) -> FeatureBuilder:
        self.embedder.fit(texts)
        self.scaler.fit(meta_features(texts))
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        return embed_cached(self.embedder, texts, self.cache_dir)

    def transform(self, texts: list[str], embeddings: np.ndarray | None = None) -> np.ndarray:
        emb = self.embed(texts) if embeddings is None else embeddings
        meta = self.scaler.transform(meta_features(texts)).astype(np.float32)
        return np.hstack([emb, meta])
