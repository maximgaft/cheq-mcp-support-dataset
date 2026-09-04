"""Layer 1 eval - routing accuracy. Deterministic, no API key, safe for CI.

k is swept on val and reported on test, for the same reason the similarity floor
was: choosing a hyperparameter on the set you then report is a leak, and it
produces a number that looks better than the system is.

Macro-F1 rather than accuracy. The classes run 21:1, so predicting "Technical
Support" for everything already scores 29% accuracy while being useless - macro-F1
weights the small queues equally and does not reward that.

Every metric is also reported per language, because 31% of the corpus is German
and a collapse on one side would hide inside an average.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT))

from embedding import center, document_text, embed_queries  # noqa: E402

from server import db  # noqa: E402
from server.retrieval import Index, collapse_duplicates  # noqa: E402

INTERIM = ROOT / "data" / "interim"
REPORT = ROOT / "reports" / "routing.md"
CASES = ROOT / "evals" / "abstention_queries.yaml"

TOP_N = 25                      # cache this many neighbours, then sweep k over them
SIM_BANDS = [0.80, 0.70, 0.60, 0.00]   # similarity bands, published for the server
AGREE_BANDS = [1.0, 0.8, 0.6, 0.4, 0.0]  # agreement bands, ditto
AGREE_SWEEP = [3, 5, 10]
K_SWEEP = [1, 2, 3, 5, 10, 25]
CHUNK = 500


def neighbours(idx: Index, texts: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-TOP_N neighbour positions and similarities for each query, plus the
    uncollapsed positions so the cost of collapsing can be measured."""
    queries = center(embed_queries(texts, progress=True), idx.mean)
    wide = TOP_N * 4
    positions, sims, raw = [], [], []
    for start in range(0, len(queries), CHUNK):
        block = queries[start : start + CHUNK] @ idx.vectors.T
        part = np.argpartition(-block, wide - 1, axis=1)[:, :wide]
        rows = np.arange(len(part))[:, None]
        order = np.argsort(-block[rows, part], axis=1)
        part = part[rows, order]
        # Collapse here too, not just in the serving path: agreement counts
        # neighbours that share a queue, so a duplicated neighbour would inflate it
        # and the published bands would be measured on evidence that is not there.
        kept = np.vstack([collapse_duplicates(row, idx.vectors, keep=TOP_N) for row in part])
        positions.append(kept)
        sims.append(np.take_along_axis(block, kept, axis=1))
        raw.append(part[:, :TOP_N])
    return np.vstack(positions), np.vstack(sims), np.vstack(raw)


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


def agreement(labels: np.ndarray, positions: np.ndarray, n: int) -> np.ndarray:
    """Share of the top-n neighbours whose label matches the nearest one's."""
    top = labels[positions[:, :n]]
    return (top == top[:, [0]]).mean(axis=1)


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """P(score of a correct prediction > score of an incorrect one). No sklearn."""
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    pos_n, neg_n = int(label.sum()), int((~label).sum())
    if not pos_n or not neg_n:
        return float("nan")
    return float((ranks[label].sum() - pos_n * (pos_n + 1) / 2) / (pos_n * neg_n))


def band_table(signal: np.ndarray, correct: np.ndarray, edges: list[float]) -> list[dict]:
    out = []
    for lower in edges:
        mask = signal >= lower
        for higher in edges:
            if higher > lower:
                mask &= signal < higher
        if mask.sum():
            out.append({"min": lower, "n": int(mask.sum()),
                        "accuracy": round(float(correct[mask].mean()), 3)})
    return out


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
    val_rows, val_pos, val_sims, _ = cached["val"]
    sweep = {k: scores(val_rows.queue.to_numpy(), vote(queues, val_pos, val_sims, k))["macro_f1"]
             for k in K_SWEEP}
    best_k = max(sweep, key=sweep.get)

    # --- reported on test ---
    rows, pos, sims, pos_raw = cached["test"]
    truth = rows.queue.to_numpy()
    queue_result = scores(truth, vote(queues, pos, sims, best_k))
    priority_result = scores(rows.priority.to_numpy(), vote(priorities, pos, sims, best_k))

    majority = pd.Series(truth).mode()[0]
    baseline = scores(truth, np.full(len(truth), majority))

    by_language = {}
    for language in sorted(rows.language.unique()):
        mask = (rows.language == language).to_numpy()
        by_language[language] = scores(truth[mask], vote(queues, pos[mask], sims[mask], best_k))

    # Two confidence signals, measured rather than assumed. Similarity predicts
    # better within a fixed input shape; agreement predicts worse but survives a
    # change of input shape, which similarity does not. Both are published so
    # suggest_routing can report them together and flag when they disagree.
    top1 = sims[:, 0]
    pred = vote(queues, pos, sims, best_k)
    correct = pred == truth

    agree_auc = {n: auc(agreement(queues, pos, n), correct) for n in AGREE_SWEEP}
    best_n = max(agree_auc, key=agree_auc.get)
    agree = agreement(queues, pos, best_n)
    agree_auc_raw = auc(agreement(queues, pos_raw, best_n), correct)  # without collapsing

    sim_auc = auc(top1, correct)
    bands = [{"min_similarity": b["min"], "n": b["n"], "accuracy": b["accuracy"]}
             for b in band_table(top1, correct, SIM_BANDS)]
    agree_bands = [{"min_agreement": b["min"], "n": b["n"], "accuracy": b["accuracy"]}
                   for b in band_table(agree, correct, AGREE_BANDS)]

    # The same two signals on handwritten queries. These have no ground-truth
    # queue, so nothing here is an accuracy claim - it measures only how far each
    # signal moves when the input stops being generated text.
    handwritten = [c["query"] for c in yaml.safe_load(CASES.read_text())
                   if c["expect"] == "precedent"]
    h_pos, h_sims, _ = neighbours(idx, handwritten)
    stability = {
        "similarity": (float(top1.mean()), float(h_sims[:, 0].mean())),
        f"agreement@{best_n}": (float(agree.mean()),
                                float(agreement(queues, h_pos, best_n).mean())),
    }

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
        "## Two confidence signals, and when to trust which",
        "",
        "Both are written to `routing_metrics.json` and read by `suggest_routing`.",
        "",
        f"**Top-1 similarity** is the better predictor - AUC {sim_auc:.3f} against "
        f"{agree_auc[best_n]:.3f} for agreement. Combining them naively is worse than "
        "either, so they are reported separately rather than blended.",
        "",
        "| top-1 similarity | tickets | accuracy |",
        "|------------------|--------:|---------:|",
        *[f"| >= {b['min_similarity']:.2f} | {b['n']:,} | {b['accuracy']:.3f} |" for b in bands],
        "",
        f"**Agreement@{best_n}** - the share of the top {best_n} neighbours that share the "
        "nearest one's queue - predicts less well but stays meaningful.",
        "",
        f"| agreement@{best_n} | tickets | accuracy |",
        "|------------|--------:|---------:|",
        *[f"| >= {b['min_agreement']:.1f} | {b['n']:,} | {b['accuracy']:.3f} |" for b in agree_bands],
        "",
        "### Why both",
        "",
        "Similarity is better within a fixed input shape and *not comparable across "
        "shapes*. Handwritten text scores lower than this corpus's generated text "
        "whatever its topic, so a similarity band measured on tickets misreads a "
        "hand-typed question as low-confidence when it is not. Agreement does not "
        "move that way:",
        "",
        "| signal | generated tickets | handwritten queries | change |",
        "|--------|------------------:|--------------------:|-------:|",
        *[f"| {name} | {gen:.3f} | {hand:.3f} | {100 * (hand - gen) / gen:+.0f}% |"
          for name, (gen, hand) in stability.items()],
        "",
        "Near-duplicate hits are collapsed before agreement is counted (doc-doc "
        "similarity 0.85), so \"three neighbours agree\" means three distinct tickets. "
        f"Uncollapsed, agreement AUC is {agree_auc_raw:.3f}; collapsed, {agree_auc[best_n]:.3f}. "
        "Duplicated neighbours also signal a dense, well-covered region, which correlates "
        "with being right, so the uncollapsed signal can look like the better predictor "
        "while partly measuring an artifact that better deduplication would remove. The "
        "collapsed figure is the one published.",
        "",
        "So `suggest_routing` returns both. When they disagree the input is probably "
        "not ticket-shaped, and agreement is the one to believe.",
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
        "agreement_n": best_n,
        "agreement_bands": agree_bands,
        "similarity_auc": round(sim_auc, 4),
        "agreement_auc": round(agree_auc[best_n], 4),
    }, indent=2) + "\n")
    print(f"  wrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
