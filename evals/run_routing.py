"""Layer 1 eval - routing accuracy. Deterministic, no API key.

Split discipline, stated once:

  val   chooses and fits. k is swept here, and the two confidence tables the
        server serves - accuracy per similarity band, accuracy per agreement
        band - are fitted here.
  test  reports and checks. The headline metrics, the paraphrase-twin
        decomposition, the confusion pairs and the handwritten-stability means
        are read from it. It also checks the val-fitted bands: the report prints
        test accuracy beside each served value, so a reader can see whether the
        calibration holds on tickets it was never fitted on. Nothing computed on
        test is served.

Choosing a hyperparameter, or fitting a calibration, on the set you then report
is a leak: it produces a number that looks better than the system is.

Macro-F1 rather than accuracy. The classes run 21:1, so predicting "Technical
Support" for everything already scores 29% accuracy while being useless - macro-F1
weights the small queues equally and does not reward that.

Every metric is also reported per language, because 31% of the corpus is German
and a collapse on one side would hide inside an average.
"""

from __future__ import annotations

import json
import re
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
K_SWEEP = [1, 2, 3, 5, 10, 25]
SIM_BANDS = [0.80, 0.70, 0.60, 0.00]   # similarity bands: fitted on val, served
AGREEMENT_N = 3
"""Neighbours the agreement share is computed over. Pinned, not swept: 3 is the
smallest n that gives a middle band (2 of 3, a majority that is not unanimous)
between unanimous and split, so the share takes exactly the three values
AGREE_BANDS enumerates. It matches the server's pre-existing default and was not
chosen on any split, so nothing about the served signal is fitted to the data it
is evaluated on."""
AGREE_BANDS = [1.0, 0.6, 0.0]           # 3 of 3, 2 of 3, 1 of 3
TWIN_JACCARD = 0.35
"""Word-set Jaccard above which a test ticket's nearest neighbour counts as a
paraphrase of it. Nearest neighbours average about 0.42; a random train ticket
about 0.03. The corpus generator produced families of reworded tickets that
exact-body dedup (stage 04) cannot see, and routing accuracy on those is lookup,
not generalisation - so the report separates the two."""
TWIN_SENSITIVITY = [0.25, TWIN_JACCARD, 0.45]
CHUNK = 500
WORD = re.compile(r"[a-zA-ZäöüßÄÖÜ]{4,}")


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
    """P(score of a correct prediction > score of an incorrect one). No sklearn.

    Ranks are averaged over ties. The agreement signal takes only three values,
    so ordinal ranks would order tied rows by position and the result would
    depend on row order."""
    ranks = pd.Series(score).rank(method="average").to_numpy()
    pos_n, neg_n = int(label.sum()), int((~label).sum())
    if not pos_n or not neg_n:
        return float("nan")
    return float((ranks[label].sum() - pos_n * (pos_n + 1) / 2) / (pos_n * neg_n))


def zscale(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    """Mean and std of each signal on the fitting split."""
    return float(a.mean()), float(a.std() or 1.0), float(b.mean()), float(b.std() or 1.0)


def zsum(a: np.ndarray, b: np.ndarray, scale: tuple[float, float, float, float]) -> np.ndarray:
    """The naive combination: standardise each signal on the fitting split's
    scale, then add. Passing val's scale to test data is what a served
    combination would do."""
    am, asd, bm, bsd = scale
    return (a - am) / asd + (b - bm) / bsd


def band_mask(signal: np.ndarray, lower: float, edges: list[float]) -> np.ndarray:
    """Rows whose signal falls in [lower, next edge above lower)."""
    mask = signal >= lower
    for higher in edges:
        if higher > lower:
            mask &= signal < higher
    return mask


def band_table(signal: np.ndarray, correct: np.ndarray, edges: list[float]) -> list[dict]:
    out = []
    for lower in edges:
        mask = band_mask(signal, lower, edges)
        if mask.sum():
            out.append({"min": lower, "n": int(mask.sum()),
                        "accuracy": round(float(correct[mask].mean()), 3)})
    return out


def sim_label(lower: float, edges: list[float]) -> str:
    """Human label for the half-open band starting at `lower`."""
    above = [e for e in edges if e > lower]
    if not above:
        return f">= {lower:.2f}"
    if lower == min(edges):
        return f"< {min(above):.2f}"
    return f"{lower:.2f} - {min(above):.2f}"


def agree_label(lower: float) -> str:
    count = max(1, round(lower * AGREEMENT_N))
    return f"{count} of {AGREEMENT_N} agree"


def jaccard(a: str, b: str) -> float:
    """Word-set overlap over words of four or more letters."""
    sa, sb = set(WORD.findall(a.lower())), set(WORD.findall(b.lower()))
    return len(sa & sb) / max(len(sa | sb), 1)


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
    train_bodies = idx.meta.body.to_numpy()

    cached = {}
    for split in ("val", "test"):
        rows = corpus[corpus.split.eq(split)].reset_index(drop=True)
        texts = [document_text(s, b) for s, b in zip(rows.subject, rows.body)]
        print(f"  embedding {len(texts):,} {split} queries...")
        cached[split] = (rows, *neighbours(idx, texts))

    # ------------------------------------------------------------------ val
    # Everything the server will apply to a live query is decided here.
    val_rows, val_pos, val_sims, val_pos_raw = cached["val"]
    val_truth = val_rows.queue.to_numpy()
    sweep = {k: scores(val_truth, vote(queues, val_pos, val_sims, k))["macro_f1"] for k in K_SWEEP}
    best_k = max(sweep, key=sweep.get)

    val_top1 = val_sims[:, 0]
    val_correct = vote(queues, val_pos, val_sims, best_k) == val_truth
    val_agree = agreement(queues, val_pos, AGREEMENT_N)
    bands = band_table(val_top1, val_correct, SIM_BANDS)            # served
    agree_bands = band_table(val_agree, val_correct, AGREE_BANDS)   # served
    scale = zscale(val_top1, val_agree)
    val_auc = {
        "top-1 similarity": auc(val_top1, val_correct),
        f"agreement@{AGREEMENT_N}": auc(val_agree, val_correct),
        f"agreement@{AGREEMENT_N}, uncollapsed": auc(
            agreement(queues, val_pos_raw, AGREEMENT_N), val_correct),
        "z-scored sum of both": auc(zsum(val_top1, val_agree, scale), val_correct),
    }

    # ----------------------------------------------------------------- test
    # Read for reporting. Nothing below feeds back into anything the server serves.
    rows, pos, sims, _ = cached["test"]
    truth = rows.queue.to_numpy()
    pred = vote(queues, pos, sims, best_k)
    correct = pred == truth
    top1 = sims[:, 0]
    agree = agreement(queues, pos, AGREEMENT_N)

    queue_result = scores(truth, pred)
    priority_result = scores(rows.priority.to_numpy(), vote(priorities, pos, sims, best_k))
    majority = pd.Series(truth).mode()[0]
    baseline = scores(truth, np.full(len(truth), majority))
    by_language = {}
    for language in sorted(rows.language.unique()):
        mask = (rows.language == language).to_numpy()
        by_language[language] = scores(truth[mask], pred[mask])

    # The check: the same bands, measured on tickets the fit never saw.
    test_bands = {b["min"]: b for b in band_table(top1, correct, SIM_BANDS)}
    test_agree_bands = {b["min"]: b for b in band_table(agree, correct, AGREE_BANDS)}
    test_auc = {
        "top-1 similarity": auc(top1, correct),
        f"agreement@{AGREEMENT_N}": auc(agree, correct),
        "z-scored sum of both": auc(zsum(top1, agree, scale), correct),
    }
    upper = [b for b in bands if b["min"] != min(SIM_BANDS)]
    sim_drift = max(abs(b["accuracy"] - test_bands[b["min"]]["accuracy"]) for b in upper)
    upper_agree = [b for b in agree_bands if b["min"] != min(AGREE_BANDS)]
    agree_drift = max(b["accuracy"] - test_agree_bands[b["min"]]["accuracy"] for b in upper_agree)

    # Paraphrase twins: is the nearest neighbour a rewording of the query? Measured
    # on the body, whatever document_text embeds, so the statistic keeps its meaning.
    bodies = rows.body.to_numpy()
    twin_j = np.array([jaccard(b, train_bodies[p]) for b, p in zip(bodies, pos[:, 0])])
    rng = np.random.default_rng(0)
    random_j = np.array([jaccard(b, train_bodies[p])
                         for b, p in zip(bodies, rng.integers(0, len(train_bodies), len(bodies)))])
    twin = twin_j >= TWIN_JACCARD
    twin_rows = [
        (f"with a paraphrase twin in the index (Jaccard >= {TWIN_JACCARD})", twin),
        ("without", ~twin),
        ("all test tickets", np.ones(len(truth), dtype=bool)),
    ]
    twin_stats = [(name, float(m.mean()), scores(truth[m], pred[m])) for name, m in twin_rows]
    sensitivity = {t: scores(truth[twin_j < t], pred[twin_j < t])["macro_f1"]
                   for t in TWIN_SENSITIVITY}
    twin_by_band = []
    for lower in SIM_BANDS:
        mask = band_mask(top1, lower, SIM_BANDS)
        if mask.sum():
            with_twin, without = mask & twin, mask & ~twin
            twin_by_band.append((
                lower, int(mask.sum()), float(twin[mask].mean()),
                float(correct[with_twin].mean()) if with_twin.any() else float("nan"),
                float(correct[without].mean()) if without.any() else float("nan"),
            ))

    # The same two signals on handwritten queries. These have no ground-truth
    # queue, so nothing here is an accuracy claim - it measures only how far each
    # signal moves when the input stops being generated text.
    handwritten = [c["query"] for c in yaml.safe_load(CASES.read_text())
                   if c["expect"] == "precedent"]
    h_pos, h_sims, _ = neighbours(idx, handwritten)
    stability = {
        "similarity": (float(top1.mean()), float(h_sims[:, 0].mean())),
        f"agreement@{AGREEMENT_N}": (float(agree.mean()),
                                     float(agreement(queues, h_pos, AGREEMENT_N).mean())),
    }

    confusion = pd.crosstab(pd.Series(truth, name="true"), pd.Series(pred, name="pred"))
    pairs = sorted(
        ((int(confusion.loc[a, b]), a, b) for a in confusion.index for b in confusion.columns
         if a != b and confusion.loc[a, b] > 0), reverse=True)
    overlap = {"Technical Support", "IT Support", "Product Support"}
    errors = truth != pred
    within = sum(1 for t, p in zip(truth[errors], pred[errors]) if t in overlap and p in overlap)

    def acc(table: dict, key: float) -> str:
        return f"{table[key]['accuracy']:.3f}" if key in table else "-"

    zgain_val = val_auc["z-scored sum of both"] - val_auc["top-1 similarity"]
    zgain_test = test_auc["z-scored sum of both"] - test_auc["top-1 similarity"]

    lines = [
        "# Routing eval",
        "",
        "Generated by `evals/run_routing.py`. Similarity-weighted k-NN over the "
        "centred embedding index. No model call, no API key.",
        "",
        f"`k` swept on **val** ({len(val_rows):,} tickets), reported on **test** "
        f"({len(rows):,} tickets, never in the index). The confidence bands the server "
        "serves are fitted on val and checked on test, below.",
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
        "## What the accuracy is made of: paraphrase twins",
        "",
        "The corpus generator produced families of reworded tickets. These are not the "
        "exact duplicates stage 04 removes - a paraphrase twin is a rewording, not a copy, "
        "and it survives that stage. Word-set Jaccard (words of 4+ letters) between each "
        f"test ticket and its nearest neighbour averages **{twin_j.mean():.3f}**, against "
        f"{random_j.mean():.3f} for a random train ticket, and {100 * twin.mean():.1f}% of "
        f"test tickets sit at or above {TWIN_JACCARD}.",
        "",
        "| test tickets | share | accuracy | macro-F1 |",
        "|---|--:|--:|--:|",
        *[f"| {name} | {100 * share:.1f}% | {s['accuracy']:.3f} | {s['macro_f1']:.3f} |"
          for name, share, s in twin_stats],
        "",
        "On the first row 1-NN is lookup: the right answer is a paraphrase of the question. "
        "The second row is the closer estimate of routing power on a ticket the archive has "
        "not seen before. The headline blends the two in whatever proportion the generator "
        "happened to produce; a real archive has its own twin rate, and the headline should "
        "be re-read against it - the 'without' row is what to expect if that rate is near zero. "
        "The split is not threshold-fragile: macro-F1 without a twin is "
        + " / ".join(f"{sensitivity[t]:.3f}" for t in TWIN_SENSITIVITY)
        + " at Jaccard cut-offs " + " / ".join(f"{t:.2f}" for t in TWIN_SENSITIVITY) + ".",
        "",
        "## Two confidence signals: fitted on val, checked on test",
        "",
        "Both tables are written to `routing_metrics.json` and read by `suggest_routing`. "
        "The served accuracy in each band is **fitted on val**; the test column is the same "
        "band measured on tickets the fit never saw, so a reader can see whether the "
        "calibration holds.",
        "",
        "| top-1 similarity | val tickets | accuracy, val (served) | accuracy, test (check) |",
        "|------------------|--------:|---------:|---------:|",
        *[f"| {sim_label(b['min'], SIM_BANDS)} | {b['n']:,} | {b['accuracy']:.3f} | {acc(test_bands, b['min'])} |"
          for b in bands],
        "",
        f"**Agreement@{AGREEMENT_N}** is the share of the top {AGREEMENT_N} neighbours that share "
        f"the nearest one's queue. {AGREEMENT_N} is pinned, not swept: it is the smallest n with "
        "a middle band (a majority that is not unanimous) between unanimous and split, so "
        "nothing about the served signal is chosen on the evaluation data.",
        "",
        f"| agreement@{AGREEMENT_N} | val tickets | accuracy, val (served) | accuracy, test (check) |",
        "|------------|--------:|---------:|---------:|",
        *[f"| {agree_label(b['min'])} | {b['n']:,} | {b['accuracy']:.3f} | {acc(test_agree_bands, b['min'])} |"
          for b in agree_bands],
        "",
        f"What the check shows: in the upper bands the served similarity accuracies hold on "
        f"test to within {sim_drift:.3f}, while the served agreement accuracies run "
        f"{agree_drift:+.3f} high. Read `expected_accuracy_by_agreement` as a ceiling.",
        "",
        "| signal | AUC, val | AUC, test |",
        "|---|--:|--:|",
        *[f"| {name} | {v:.3f} | {test_auc[name]:.3f} |" if name in test_auc else f"| {name} | {v:.3f} | - |"
          for name, v in val_auc.items()],
        "",
        f"Top-1 similarity is the better predictor. A z-scored sum of the two adds {zgain_val:+.3f} "
        f"AUC on val and {zgain_test:+.3f} on test, and would hide which signal moved, so they "
        "are reported separately. They also fail differently, below.",
        "",
        "The twins shape the similarity bands. `expected_accuracy` is therefore partly a "
        "measure of *whether a paraphrase of this ticket is already in the index* - which is "
        "what a precedent tool should measure, but a reader should know that is what the "
        "number means.",
        "",
        "| top-1 similarity | test tickets | share with twin | accuracy with twin | accuracy without |",
        "|---|--:|--:|--:|--:|",
        *[f"| {sim_label(lo, SIM_BANDS)} | {n:,} | {100 * share:.0f}% | {a_t:.3f} | {a_n:.3f} |"
          for lo, n, share, a_t, a_n in twin_by_band],
        "",
        "### Why both",
        "",
        "Similarity is better within a fixed input shape and *not comparable across "
        "shapes*. Handwritten text scores lower than this corpus's generated text "
        "whatever its topic, so a similarity band measured on tickets misreads a "
        "hand-typed question as low-confidence when it is not. Agreement moves the "
        "other way:",
        "",
        f"| signal | generated tickets (test, n={len(rows):,}) | handwritten queries (n={len(handwritten)}) | change |",
        "|--------|------------------:|--------------------:|-------:|",
        *[f"| {name} | {gen:.3f} | {hand:.3f} | {100 * (hand - gen) / gen:+.0f}% |"
          for name, (gen, hand) in stability.items()],
        "",
        "Near-duplicate hits are collapsed before agreement is counted (doc-doc "
        "similarity 0.85), so \"three neighbours agree\" means three distinct tickets. "
        f"On val, uncollapsed agreement scores AUC {val_auc[f'agreement@{AGREEMENT_N}, uncollapsed']:.3f} "
        f"against {val_auc[f'agreement@{AGREEMENT_N}']:.3f} collapsed. Duplicated neighbours also "
        "signal a dense, well-covered region, which correlates with being right, so the "
        "uncollapsed signal can look like the better predictor while partly measuring an "
        "artifact that better deduplication would remove. The collapsed figure is the one "
        "published.",
        "",
        "So `suggest_routing` returns both. When they disagree, the neighbours are listed so "
        "the caller can look rather than guess.",
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
        "fitted_on": "val",
        "k": best_k,
        "agreement_n": AGREEMENT_N,
        "reliability_bands": [{"min_similarity": b["min"], "n": b["n"], "accuracy": b["accuracy"]}
                              for b in bands],
        "agreement_bands": [{"min_agreement": b["min"], "n": b["n"], "accuracy": b["accuracy"]}
                            for b in agree_bands],
        "n_val": len(val_rows),
        "n_test": len(rows),
        "queue_macro_f1": round(queue_result["macro_f1"], 4),
        "queue_accuracy": round(queue_result["accuracy"], 4),
        "priority_macro_f1": round(priority_result["macro_f1"], 4),
        "baseline_macro_f1": round(baseline["macro_f1"], 4),
        "by_language": {k: round(v["macro_f1"], 4) for k, v in by_language.items()},
        "auc": {"val": {k: round(v, 4) for k, v in val_auc.items()},
                "test": {k: round(v, 4) for k, v in test_auc.items()}},
        "twin_jaccard": TWIN_JACCARD,
        "twin_share_test": round(float(twin.mean()), 4),
        "macro_f1_without_twin": round(twin_stats[1][2]["macro_f1"], 4),
        "macro_f1_without_twin_by_cutoff": {str(t): round(v, 4) for t, v in sensitivity.items()},
    }, indent=2) + "\n")
    print(f"  wrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
