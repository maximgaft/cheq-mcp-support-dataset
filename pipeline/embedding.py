"""Embedding configuration shared by the build pipeline and the MCP server.

E5 models are asymmetric: indexed documents must be embedded as "passage: ..."
and searches as "query: ...". Getting that backwards - or dropping the prefixes -
costs retrieval quality silently, with no error anywhere. Both sides import from
here so the strings exist once.

Vectors are L2-normalised, so cosine similarity is a plain dot product.
"""

from __future__ import annotations

import functools

import numpy as np

MODEL_NAME = "intfloat/multilingual-e5-small"
DIM = 384


@functools.lru_cache(maxsize=1)
def load_model():
    import torch
    from sentence_transformers import SentenceTransformer

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return SentenceTransformer(MODEL_NAME, device=device)


def document_text(subject: str | None, body: str) -> str:
    """Subject is a useful summary when present; it is missing on ~8.6% of tickets."""
    if isinstance(subject, str) and subject.strip():
        return f"{subject.strip()}\n\n{body}"
    return body


def embed_documents(texts: list[str], batch_size: int = 64, progress: bool = False) -> np.ndarray:
    model = load_model()
    return model.encode(
        [f"passage: {t}" for t in texts],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=progress,
        convert_to_numpy=True,
    ).astype(np.float32)


def center(vectors: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Subtract the corpus-mean direction, then re-normalise to unit length.

    Every embedding in this corpus shares a large common component - business
    prose, politely worded, about a product problem - and that shared direction
    dominates the cosine. Raw, two unrelated tickets score 0.82; centred, they
    score 0.00, which is what "unrelated" should look like.

    Works on a single vector or a batch. Queries MUST be centred with the same
    mean as the index, or similarity degrades with no error.
    """
    x = vectors - mean
    return (x / np.linalg.norm(x, axis=-1, keepdims=True)).astype(np.float32)


def embed_queries(texts: list[str], batch_size: int = 64, progress: bool = False) -> np.ndarray:
    """Batch form of embed_query, for evaluation over thousands of queries."""
    model = load_model()
    return model.encode(
        [f"query: {t}" for t in texts],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=progress,
        convert_to_numpy=True,
    ).astype(np.float32)


def embed_query(text: str) -> np.ndarray:
    model = load_model()
    return model.encode(
        f"query: {text}",
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
