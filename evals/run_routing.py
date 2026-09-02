"""Layer 1 eval - routing accuracy. Deterministic, no API key, safe for CI.

k is swept on val and reported on test, for the same reason the similarity floor
was: choosing a hyperparameter on the set you then report is a leak, and it
produces a number that looks better than the system is.

Macro-F1 rather than accuracy. The classes run 21:1, so predicting "Technical
Support" for everything already scores 29% accuracy while being useless - macro-F1
weights the small queues equally and does not reward that.

Every metric is also reported per language, because 41% of the corpus is German
and a collapse on one side would hide inside an average.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT))

from embedding import center, document_text, embed_queries  # noqa: E402

from server import db  # noqa: E402
from server.retrieval import Index  # noqa: E402

INTERIM = ROOT / "data" / "interim"
REPORT = ROOT / "reports" / "routing.md"

TOP_N = 25                      # cache this many neighbours, then sweep k over them
BANDS = [0.80, 0.70, 0.60, 0.00]  # reliability bands, published for the server to read
K_SWEEP = [1, 2, 3, 5, 10, 25]
CHUNK = 500


def neighbours(idx: Index, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Top-TOP_N neighbour positions and similarities for each query."""
    queries = center(embed_queries(texts, progress=True), idx.mean)
    positions, sims = [], []
    for start in range(0, len(queries), CHUNK):
        block = queries[start : start + CHUNK] @ idx.vectors.T
        part = np.argpartition(-block, TOP_N - 1, axis=1)[:, :TOP_N]
        rows = np.arange(len(part))[:, None]
        order = np.argsort(-block[rows, part], axis=1)
        positions.append(part[rows, order])
        sims.append(block[rows, part][rows, order])
    return np.vstack(positions), np.vstack(sims)


def vote(labels: np.ndarray, positions: np.ndarray, sims: np.ndarray, k: int) -> np.ndarray:
    """Similarity-weighted vote over the first k neighbours."""
    out = []
    for i in range(len(positions)):
        weights: dict[str, float] = {}
        for j in range(k):
            label = labels[positions[i, j]]
            weights[label] = weights.get(label, 0.0) + max(float(sims[i, j]), 0.0)
        out.append(max(weights, key=weights.get))
    return np.array(out)


def scores(truth: np.ndarray, pred: np.ndarray) -> dict:
    """Per-class precision/recall/F1 and the macro average, without sklearn."""
    per_class = {}
    for label in sorted(set(truth)):
        tp = int(((pred == label) & (truth == label)).sum())
        fp = int(((pred == label) & (truth != label)).sum())
        fn = int(((pred != label) & (truth == label)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"support": tp + fn, "precision": precision,
                            "recall": recall, "f1": f1}
    return {
        "accuracy": float((truth == pred).mean()),
        "macro_f1": float(np.mean([m["f1"] for m in per_class.values()])),
        "per_class": per_class,
    }


def main() -> None:
    con = db.connect(INTERIM / "08_tickets.duckdb")
    idx = Index(INTERIM / "06_index.npz", INTERIM / "07_thresholds.json", con)
    corpus = pd.read_parquet(INTERIM / "05_split.parquet")
    queues = idx.meta.queue.to_numpy()
    priorities = idx.meta.priority.to_numpy()

    cached = {}
    for split in ("val", "test"):
        rows = corpus[corpus.split.eq(split)].reset_index(drop=True)
        texts = [document_text(s, b) for s, b in zip(rows.subject, rows.body)]
        print(f"  embedding {len(texts):,} {split} queries...")
        cached[split] = (rows, *neighbours(idx, texts))

    # --- k chosen on val ---
    val_rows, val_pos, val_sims = cached["val"]
    sweep = {k: scores(val_rows.queue.to_numpy(), vote(queues, val_pos, val_sims, k))["macro_f1"]
             for k in K_SWEEP}
    best_k = max(sweep, key=sweep.get)

    # --- reported on test ---
    rows, pos, sims = cached["test"]
    truth = rows.queue.to_numpy()
    queue_result = scores(truth, vote(queues, pos, sims, best_k))
    priority_result = scores(rows.priority.to_numpy(), vote(priorities, pos, sims, best_k))

    majority = pd.Series(truth).mode()[0]
    baseline = scores(truth, np.full(len(truth), majority))

    by_language = {}
    for language in sorted(rows.language.unique()):
        mask = (rows.language == language).to_numpy()
        by_language[language] = scores(truth[mask], vote(queues, pos[mask], sims[mask], best_k))

    # Accuracy is strongly predictable from the top-1 similarity. Publishing these
    # bands lets suggest_routing report how much to trust a prediction, instead of
    # returning a bare label.
    top1 = sims[:, 0]
    pred = vote(queues, pos, sims, best_k)
    bands = []
    for lower in BANDS:
        mask = top1 >= lower
        for higher in BANDS:
            if higher > lower:
                mask &= top1 < higher
        if mask.sum():
            bands.append({"min_similarity": lower, "n": int(mask.sum()),
                          "accuracy": round(float((pred[mask] == truth[mask]).mean()), 3)})

    confusion = pd.crosstab(pd.Series(truth, name="true"), pd.Series(pred, name="pred"))
    pairs = sorted(
        ((int(confusion.loc[a, b]), a, b) for a in confusion.index for b in confusion.columns
         if a != b and confusion.loc[a, b] > 0), reverse=True)
    overlap = {"Technical Support", "IT Support", "Product Support"}
    errors = truth != pred
    within = sum(1 for t, p in zip(truth[errors], pred[errors]) if t in overlap and p in overlap)

    lines = [
        "# Routing eval",
        "",
        "Generated by `evals/run_routing.py`. Similarity-weighted k-NN over the "
        "centred embedding index. No model call, no API key.",
        "",
        f"`k` swept on **val** ({len(val_rows):,} tickets), reported on **test** "
        f"({len(rows):,} tickets, never in the index).",
        "",
        "| k | val macro-F1 |",
        "|--:|-------------:|",
        *[f"| {k}{' **(chosen)**' if k == best_k else ''} | {sweep[k]:.3f} |" for k in K_SWEEP],
        "",
        "## Test results",
        "",
        "| target | accuracy | macro-F1 |",
        "|--------|---------:|---------:|",
        f"| queue (10 classes) | {queue_result['accuracy']:.3f} | **{queue_result['macro_f1']:.3f}** |",
        f"| priority (3 classes) | {priority_result['accuracy']:.3f} | {priority_result['macro_f1']:.3f} |",
        f"| baseline: always \"{majority}\" | {baseline['accuracy']:.3f} | {baseline['macro_f1']:.3f} |",
        "",
        "Macro-F1 is the headline, not accuracy: the majority-class baseline reaches "
        f"{baseline['accuracy']:.0%} accuracy on a 21:1 class distribution while being useless, "
        f"and scores {baseline['macro_f1']:.3f} macro-F1.",
        "",
        "## Per language",
        "",
        "| language | tickets | accuracy | macro-F1 |",
        "|----------|--------:|---------:|---------:|",
        *[f"| {lang} | {int((rows.language == lang).sum()):,} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} |"
          for lang, m in by_language.items()],
        "",
        "## How much to trust a prediction",
        "",
        "Accuracy tracks the top-1 similarity closely, so the score is a usable "
        "confidence signal rather than a diagnostic. These bands are written to "
        "`routing_metrics.json` and read by `suggest_routing`.",
        "",
        "| top-1 similarity | tickets | accuracy |",
        "|------------------|--------:|---------:|",
        *[f"| >= {b['min_similarity']:.2f} | {b['n']:,} | {b['accuracy']:.3f} |" for b in bands],
        "",
        f"Near-duplicate matching contributes little: only {100 * (top1 >= 0.80).mean():.1f}% of "
        f"test tickets have a neighbour above 0.80, and excluding all of them macro-F1 is "
        f"{scores(truth[top1 < 0.80], vote(queues, pos[top1 < 0.80], sims[top1 < 0.80], best_k))['macro_f1']:.3f} "
        f"against {queue_result['macro_f1']:.3f} overall. Exact-duplicate removal was necessary "
        "but this corpus also contains paraphrase pairs, so a similarity-based dedup would be "
        "the next step on a real archive.",
        "",
        "## Where the errors are",
        "",
        f"{int(errors.sum()):,} of {len(truth):,} test tickets are misrouted. "
        f"{within} of those errors ({100 * within / errors.sum():.0f}%) are *within* "
        "Technical Support / IT Support / Product Support - three queues whose tickets "
        "overlap semantically, and which a human triager would also confuse. The ceiling "
        "here is label ambiguity, not model capacity.",
        "",
        "| confused | count |",
        "|----------|------:|",
        *[f"| {a} -> {b} | {n} |" for n, a, b in pairs[:6]],
        "",
        "## Per queue (test)",
        "",
        "| queue | support | precision | recall | F1 |",
        "|-------|--------:|----------:|-------:|---:|",
        *[f"| {label} | {m['support']} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |"
          for label, m in sorted(queue_result["per_class"].items(),
                                 key=lambda kv: -kv[1]["f1"])],
        "",
    ]

    print("\n" + "\n".join(lines[6:]))
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines))
    (INTERIM / "routing_metrics.json").write_text(json.dumps({
        "k": best_k,
        "queue_macro_f1": round(queue_result["macro_f1"], 4),
        "queue_accuracy": round(queue_result["accuracy"], 4),
        "priority_macro_f1": round(priority_result["macro_f1"], 4),
        "baseline_macro_f1": round(baseline["macro_f1"], 4),
        "by_language": {k: round(v["macro_f1"], 4) for k, v in by_language.items()},
        "n_test": len(rows),
        "reliability_bands": bands,
    }, indent=2) + "\n")
    print(f"  wrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
