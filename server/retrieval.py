"""Vector search over the ticket index, with the abstention decision attached.

Search is a brute-force dot product against 31,625 x 384 centred vectors: one BLAS
matrix-vector product, ~12M multiply-adds, measured at 0.43 ms and 2,300 queries
a second. It stays under 10 ms out to about a million vectors.

Exact, not approximate, and that matters more than the speed. An ANN index runs
at 95-99% recall, so a few percent of the time it substitutes a worse neighbour.
Harmless in most applications - but this project reports measured retrieval and
routing numbers, and approximation would add a second error source that could not
be separated from the first. Every number in reports/ is then attributable to the
embeddings alone.

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
from pipeline.embedding import center, embed_query  # noqa: E402

NEAR_DUPLICATE = 0.85
"""Doc-doc similarity above which two hits are the same ticket reworded.

The corpus contains paraphrase pairs that exact-body dedup cannot catch. A known
pair measures 0.874; the 90th percentile of similarity between a query's top-10
hits is 0.812. 0.85 is the highest cut that still catches the known pair, and it
removes about one hit in five.

This matters twice. It stops k=5 returning three distinct precedents, and it stops
neighbour agreement counting one ticket twice - agreement reads as corroboration,
so a duplicated neighbour inflates it with evidence that is not there.
"""


def collapse_duplicates(positions: np.ndarray, vectors: np.ndarray,
                        keep: int, threshold: float = NEAR_DUPLICATE) -> np.ndarray:
    """Greedily keep hits that are not near-duplicates of an already-kept hit.

    Always returns exactly `keep` entries (or every input, if there are fewer):
    if collapsing leaves too few, the best-ranked rejects are appended back so the
    caller's array shape is predictable.
    """
    chosen, rejected = [0], []
    for i in range(1, len(positions)):
        if len(chosen) >= keep:
            rejected.append(i)
        elif all(float(vectors[positions[i]] @ vectors[positions[j]]) < threshold
                 for j in chosen):
            chosen.append(i)
        else:
            rejected.append(i)
    if len(chosen) < keep:
        chosen += rejected[: keep - len(chosen)]
        chosen.sort()
    return positions[chosen[:keep]]


META_COLUMNS = ["ticket_id", "queue", "priority", "type", "language", "subject",
                "body", "answer", "reply_state"]


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
        self.agreement_n = 3
        self.bands: list[dict] = []
        self.agreement_bands: list[dict] = []
        if routing_metrics_path is not None and routing_metrics_path.exists():
            metrics = json.loads(routing_metrics_path.read_text())
            self.vote_k = int(metrics.get("k", 1))
            self.agreement_n = int(metrics.get("agreement_n", 3))
            self.bands = sorted(metrics.get("reliability_bands", []),
                                key=lambda b: -b["min_similarity"])
            self.agreement_bands = sorted(metrics.get("agreement_bands", []),
                                          key=lambda b: -b["min_agreement"])

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

        # Over-fetch, then collapse near-duplicates down to k.
        available = int(np.isfinite(sims).sum())
        wide = min(max(k * 4, k), available)
        order = np.argpartition(-sims, wide - 1)[:wide] if wide < len(sims) else np.arange(len(sims))
        order = order[np.argsort(-sims[order])]
        order = collapse_duplicates(order, self.vectors, keep=k)

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
                "reply_state": None if pd.isna(row.reply_state) else row.reply_state,
            })
        return hits, (hits[0]["similarity"] if hits else float("nan"))

    def expected_accuracy(self, similarity: float) -> float | None:
        """Measured routing accuracy for predictions at this similarity."""
        for band in self.bands:
            if similarity >= band["min_similarity"]:
                return band["accuracy"]
        return None

    def expected_accuracy_from_agreement(self, share: float) -> float | None:
        for band in self.agreement_bands:
            if share >= band["min_agreement"]:
                return band["accuracy"]
        return None

    def route(self, text: str, show_k: int = 5) -> dict:
        """Vote over the measured-best number of neighbours, show more for context.

        The vote uses self.vote_k, which evals/run_routing.py chose on the val
        split. show_k only controls how many neighbours come back for inspection -
        widening it would not change the prediction, and widening the *vote*
        measurably degrades it (val macro-F1 0.708 at k=1 falls to 0.297 at k=25,
        because a larger neighbourhood floods the small queues with the majority
        class).
        """
        hits, top = self.search(text, max(show_k, self.vote_k, self.agreement_n))
        if not hits:
            return {"queue": None, "priority": None, "neighbours": []}

        voters = hits[: self.vote_k]
        nearest = hits[: self.agreement_n]
        share = sum(h["queue"] == nearest[0]["queue"] for h in nearest) / len(nearest)

        def vote(field: str) -> tuple[str, float]:
            weights: dict[str, float] = {}
            for hit in voters:
                weights[hit[field]] = weights.get(hit[field], 0.0) + max(hit["similarity"], 0.0)
            total = sum(weights.values()) or 1.0
            winner = max(weights, key=weights.get)
            return winner, weights[winner] / total

        queue, _ = vote("queue")
        priority, _ = vote("priority")
        by_similarity = self.expected_accuracy(top)
        by_agreement = self.expected_accuracy_from_agreement(share)

        # Three states, checked in order.
        #
        # Both confidence figures were measured on inputs that have a real
        # neighbourhood. Below the abstention floor there is none, and agreement
        # becomes actively misleading - it ignores the base rate, so three
        # neighbours from Technical Support (29% of the corpus) "agree" on a query
        # with no content in it. The floor is the gate; the two signals only refine
        # what sits above it.
        #
        # Above the floor, similarity is the better predictor but is not comparable
        # across input shapes: handwritten text scores ~28% lower than this corpus's
        # generated text whatever its topic, while agreement does not fall. So a wide
        # gap between them says more about the input than about the prediction.
        guidance = None
        if top < self.floor:
            by_similarity = by_agreement = None
            guidance = (
                f"The nearest ticket scores {top:.3f}, below the {self.floor} abstention "
                "floor - there is no usable neighbourhood here, so neither confidence "
                "figure applies and both are withheld. Route this to a human. Note that "
                "neighbours can still appear to agree at this range: agreement ignores "
                "how common a queue is, and the largest queue is 29% of the corpus."
            )
        elif by_similarity is not None and by_agreement is not None:
            if by_agreement - by_similarity > 0.15:
                guidance = (
                    f"The two signals disagree ({by_similarity:.2f} by similarity, "
                    f"{by_agreement:.2f} by agreement). That gap usually means the input "
                    "is not ticket-shaped - a hand-typed question rather than a real "
                    "ticket - which depresses similarity without affecting agreement. "
                    "Trust the agreement figure here."
                )
            elif by_similarity - by_agreement > 0.15:
                guidance = (
                    f"Close nearest match ({top:.3f}) but the neighbours disagree "
                    f"({share:.0%} share a queue). The ticket sits between queues; treat "
                    f"{by_agreement:.2f} as the honest number and check the neighbours."
                )

        return {
            "queue": queue,
            "priority": priority,
            "top_similarity": round(top, 3),
            "expected_accuracy": by_similarity,
            "neighbour_agreement": round(share, 3),
            "expected_accuracy_by_agreement": by_agreement,
            "guidance": guidance,
            "voted_over": self.vote_k,
            "neighbours": [
                {"ticket_id": h["ticket_id"], "queue": h["queue"],
                 "priority": h["priority"], "similarity": h["similarity"],
                 "voted": i < self.vote_k}
                for i, h in enumerate(hits)
            ],
        }
