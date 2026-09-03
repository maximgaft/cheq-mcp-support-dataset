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
    """Embed the body alone. `subject` is accepted and deliberately ignored.

    Concatenating subject + body looks obviously right and measurably is not.
    Subject is missing on ~10% of tickets, so concatenating it creates two
    populations of document that do not compete fairly: a document matches best
    when its subject-presence matches the query's. The effect was large - tickets
    with no subject are 10.3% of the index but were 27.0% of top-1 hits and 36.8%
    of the top 3, over-represented 2.6x for a reason that carries no information.

    Dropping the subject removes it (27.0% -> 11.8%, against a 10.3% base rate)
    and costs 0.006 macro-F1 on routing, which is noise - accuracy is fractionally
    better. Subject is still returned with every hit for a human to read; it is
    only kept out of the vector.

    The signature keeps `subject` so callers stay unchanged and the omission is
    visible here rather than scattered across call sites.
    """
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
