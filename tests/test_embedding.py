"""The embedding contract: E5 prefixes, centring, body-only documents.

The model is replaced by a fake that records what it was asked to encode. Getting
the prefixes wrong costs retrieval quality silently, which is why this is pinned.
"""

import numpy as np

import pipeline.embedding as emb


class FakeModel:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append(texts)
        n = 1 if isinstance(texts, str) else len(texts)
        out = np.full((n, emb.DIM), 1 / np.sqrt(emb.DIM), dtype=np.float32)
        return out[0] if isinstance(texts, str) else out


def test_e5_prefixes_are_applied(monkeypatch):
    fake = FakeModel()
    monkeypatch.setattr(emb, "load_model", lambda: fake)
    emb.embed_documents(["a", "b"])
    emb.embed_queries(["c"])
    emb.embed_query("d")
    assert fake.calls == [["passage: a", "passage: b"], ["query: c"], "query: d"]


def test_center_subtracts_then_renormalises():
    mean = np.array([1.0, 0.0], dtype=np.float32)
    batch = emb.center(np.array([[3.0, 4.0], [1.0, 1.0]], dtype=np.float32), mean)
    assert batch.dtype == np.float32 and np.allclose(np.linalg.norm(batch, axis=1), 1.0)
    assert np.allclose(batch[0], [2 / np.sqrt(20), 4 / np.sqrt(20)], atol=1e-6)   # (3,4)-(1,0) = (2,4), unit
    single = emb.center(np.array([3.0, 4.0], dtype=np.float32), mean)
    assert single.shape == (2,) and np.isclose(np.linalg.norm(single), 1.0)


def test_document_text_embeds_the_body_only():
    assert emb.document_text("Subject line", "Body text") == "Body text"
    assert emb.document_text(None, "Body text") == "Body text"
