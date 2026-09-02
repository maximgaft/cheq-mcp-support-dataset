"""Vector search over the ticket index, with the abstention decision attached.

Search is a brute-force dot product against 31,795 x 384 centred vectors - about
15 million floating-point operations, well under a millisecond. No approximate
index, so no recall loss to confound the retrieval numbers, and no vector
database to run.

Self-retrieval is structurally impossible rather than guarded against: the index
holds only `split == "train"` tickets, so a val or test ticket used as a query
cannot find itself. That is what makes the measured numbers mean anything.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from embedding import center, document_text, embed_query  # noqa: E402

META_COLUMNS = ["ticket_id", "queue", "priority", "type", "language", "subject", "body", "answer"]


class Index:
    """The embedding index plus the ticket metadata aligned to it."""

    def __init__(self, npz_path: Path, thresholds_path: Path, con,
                 routing_metrics_path: Path | None = None) -> None:
        blob = np.load(npz_path, allow_pickle=True)
        self.ticket_id = blob["ticket_id"]
        self.vectors = blob["vectors"]
        self.mean = blob["mean"]
        self.floor = float(json.loads(thresholds_path.read_text())["similarity_floor"])

        # k and the reliability bands are measured by evals/run_routing.py, not
        # guessed here. Without that file we fall back to k=1 (the measured
        # optimum) and report no calibration rather than inventing one.
        self.vote_k = 1
        self.bands: list[dict] = []
        if routing_metrics_path is not None and routing_metrics_path.exists():
            metrics = json.loads(routing_metrics_path.read_text())
            self.vote_k = int(metrics.get("k", 1))
            self.bands = sorted(
                metrics.get("reliability_bands", []),
                key=lambda b: -b["min_similarity"],
            )

        rows = con.execute(f"SELECT {', '.join(META_COLUMNS)} FROM tickets").df()
        self.meta = rows.set_index("ticket_id").reindex(self.ticket_id).reset_index()
        missing = int(self.meta.queue.isna().sum())
        if missing:
            raise RuntimeError(
                f"{missing} indexed tickets are absent from the database - "
                "the index and the database were built from different data"
            )

    def embed(self, text: str) -> np.ndarray:
        return center(embed_query(text), self.mean)

    def search(self, text: str, k: int, *, queue: str | None = None,
               language: str | None = None) -> tuple[list[dict], float]:
        """Return the k nearest tickets and the top similarity."""
        sims = self.vectors @ self.embed(text)

        mask = None
        if queue:
            mask = (self.meta.queue == queue).to_numpy()
        if language:
            lang_mask = (self.meta.language == language).to_numpy()
            mask = lang_mask if mask is None else (mask & lang_mask)
        if mask is not None:
            if not mask.any():
                return [], float("nan")
            sims = np.where(mask, sims, -np.inf)

        k = min(k, int(np.isfinite(sims).sum()))
        order = np.argpartition(-sims, k - 1)[:k] if k < len(sims) else np.arange(len(sims))
        order = order[np.argsort(-sims[order])]

        hits = []
        for position in order:
            row = self.meta.iloc[position]
            hits.append({
                "ticket_id": row.ticket_id,
                "similarity": round(float(sims[position]), 3),
                "queue": row.queue,
                "priority": row.priority,
                "type": row.type,
                "language": row.language,
                "subject": None if pd.isna(row.subject) else row.subject,
                "body": row.body,
                "answer": row.answer,
            })
        return hits, (hits[0]["similarity"] if hits else float("nan"))

    def top1(self, text: str) -> float:
        return float((self.vectors @ self.embed(text)).max())

    def expected_accuracy(self, similarity: float) -> float | None:
        """Measured routing accuracy for predictions at this similarity."""
        for band in self.bands:
            if similarity >= band["min_similarity"]:
                return band["accuracy"]
        return None

    def route(self, text: str, show_k: int = 5) -> dict:
        """Vote over the measured-best number of neighbours, show more for context.

        The vote uses self.vote_k, which evals/run_routing.py chose on the val
        split. show_k only controls how many neighbours come back for inspection -
        widening it would not change the prediction, and widening the *vote*
        measurably degrades it (macro-F1 0.691 at k=1 falls to 0.313 at k=25,
        because a larger neighbourhood floods the small queues with the majority
        class).
        """
        hits, top = self.search(text, max(show_k, self.vote_k))
        if not hits:
            return {"queue": None, "priority": None, "neighbours": []}

        voters = hits[: self.vote_k]

        def vote(field: str) -> tuple[str, float]:
            weights: dict[str, float] = {}
            for hit in voters:
                weights[hit[field]] = weights.get(hit[field], 0.0) + max(hit["similarity"], 0.0)
            total = sum(weights.values()) or 1.0
            winner = max(weights, key=weights.get)
            return winner, weights[winner] / total

        queue, queue_share = vote("queue")
        priority, priority_share = vote("priority")
        return {
            "queue": queue,
            "priority": priority,
            "top_similarity": round(top, 3),
            "expected_accuracy": self.expected_accuracy(top),
            "voted_over": self.vote_k,
            "vote_share": round(queue_share, 3),
            "priority_vote_share": round(priority_share, 3),
            "neighbours": [
                {"ticket_id": h["ticket_id"], "queue": h["queue"],
                 "priority": h["priority"], "similarity": h["similarity"],
                 "voted": i < self.vote_k}
                for i, h in enumerate(hits)
            ],
        }


def query_text(subject, body) -> str:
    """Build the query string the same way the index built its documents."""
    return document_text(subject, body)
