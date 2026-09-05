"""Vector search and the routing decision, on a six-ticket index built in
tmp_path. Index.embed is replaced so no model loads; the geometry is chosen so
each test pins one rule.

Every ticket vector is unit(cos, axis): cosine `cos` to e0, leaning into its own
axis, so the query q(a) = a*e0 + sqrt(1-a^2)*e6 scores cos*a against every ticket
and no two distinct tickets look like near-duplicates.
"""

import json

import duckdb
import numpy as np
import pytest

from server import db
from server.retrieval import GUIDANCE_GAP, Index, collapse_duplicates

DIM = 8
TICKETS = [
    # id,   cos,  axis, queue,               priority, language
    ("t0", 0.90, 1, "Billing and Payments", "high",   "en"),
    ("t1", 0.88, 2, "Billing and Payments", "high",   "en"),
    ("t2", 0.70, 3, "Billing and Payments", "low",    "de"),
    ("t3", 0.65, 4, "Technical Support",    "medium", "en"),
    ("t4", 0.30, 5, "IT Support",           "low",    "en"),
    ("t5", 0.899, 1, "Billing and Payments", "high",  "en"),   # near twin of t0: doc-doc cosine ~1.0, ranks just below it
]
BANDS = {
    "k": 1, "agreement_n": 3,
    "reliability_bands": [{"min_similarity": 0.8, "accuracy": 0.97},
                          {"min_similarity": 0.6, "accuracy": 0.67},
                          {"min_similarity": 0.0, "accuracy": 0.46}],
    "agreement_bands": [{"min_agreement": 1.0, "accuracy": 0.92},
                        {"min_agreement": 0.6, "accuracy": 0.80},
                        {"min_agreement": 0.0, "accuracy": 0.63}],
}


def unit(cos: float, axis: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    v[0], v[axis] = cos, np.sqrt(1 - cos ** 2)
    return v


def q(a: float) -> np.ndarray:
    return unit(a, 6)


def build(tmp_path, drop=()):
    keep = [t for t in TICKETS if t[0] not in drop]
    np.savez(tmp_path / "index.npz",
             ticket_id=np.asarray([t[0] for t in TICKETS], dtype=str),
             vectors=np.stack([unit(t[1], t[2]) for t in TICKETS]),
             mean=np.zeros(DIM, dtype=np.float32))
    (tmp_path / "thresholds.json").write_text(json.dumps({"similarity_floor": 0.47}))
    (tmp_path / "metrics.json").write_text(json.dumps(BANDS))
    writer = duckdb.connect(str(tmp_path / "t.duckdb"))
    writer.execute("CREATE TABLE tickets (ticket_id VARCHAR, queue VARCHAR, priority VARCHAR, "
                   "type VARCHAR, language VARCHAR, subject VARCHAR, body VARCHAR, answer VARCHAR, "
                   "reply_state VARCHAR)")
    for tid, _, _, queue, priority, language in keep:
        writer.execute("INSERT INTO tickets VALUES (?, ?, ?, 'Incident', ?, 'subj', 'body', 'ans', 'resolved')",
                       [tid, queue, priority, language])
    writer.close()
    idx = Index(tmp_path / "index.npz", tmp_path / "thresholds.json",
                db.connect(tmp_path / "t.duckdb"), tmp_path / "metrics.json")
    idx.embed = lambda text: q(float(text))   # the "text" is the cosine to e0
    return idx


@pytest.fixture
def idx(tmp_path):
    return build(tmp_path)


def test_without_metrics_file_falls_back(tmp_path):
    idx = build(tmp_path)
    (tmp_path / "metrics.json").unlink()
    bare = Index(tmp_path / "index.npz", tmp_path / "thresholds.json",
                 db.connect(tmp_path / "t.duckdb"), tmp_path / "metrics.json")
    bare.embed = idx.embed
    r = bare.route("1.0")
    assert r["voted_over"] == 1 and r["queue"] == "Billing and Payments"
    assert r["expected_accuracy"] is None and r["expected_accuracy_by_agreement"] is None
    assert r["guidance"] is None


def test_search_ranks_by_similarity_and_collapses_the_twin(idx):
    hits, top = idx.search("1.0", k=3)
    assert [h["ticket_id"] for h in hits] == ["t0", "t1", "t2"]   # t5 collapsed into t0
    assert top == hits[0]["similarity"] == pytest.approx(0.9)      # full precision, not rounded


def test_decisions_use_full_precision_not_the_rounded_display(idx):
    r = idx.route(str(0.4696 / 0.9))   # top 0.4696 displays as 0.470 but is below the 0.47 floor
    assert r["top_similarity"] == 0.47 and r["expected_accuracy"] is None
    assert "0.4696" in r["guidance"] and "0.470" not in r["guidance"]   # the sentence stays true
    r = idx.route(str(0.7996 / 0.9))   # top 0.7996 displays as 0.800 but sits in the 0.6-0.8 band
    assert r["top_similarity"] == 0.8 and r["expected_accuracy"] == 0.67


def test_search_filters(idx):
    assert [h["ticket_id"] for h in idx.search("1.0", k=5, queue="Technical Support")[0]] == ["t3"]
    assert [h["ticket_id"] for h in idx.search("1.0", k=5, language="de")[0]] == ["t2"]
    hits, top = idx.search("1.0", k=5, queue="No Such Queue")
    assert hits == [] and np.isnan(top)


def test_route_confident_no_guidance(idx):
    r = idx.route("1.0")
    assert (r["queue"], r["priority"]) == ("Billing and Payments", "high")
    assert r["expected_accuracy"] == 0.97 and r["neighbour_agreement"] == 1.0
    assert r["expected_accuracy_by_agreement"] == 0.92
    assert r["guidance"] is None
    assert r["voted_over"] == 1 and [n["voted"] for n in r["neighbours"]] == [True, False, False, False, False]


def test_route_divergence_is_named_not_diagnosed(idx):
    r = idx.route("0.78")   # top 0.702: band 0.6-0.8, but the top three agree
    assert r["expected_accuracy"] == 0.67 and r["expected_accuracy_by_agreement"] == 0.92
    assert abs(r["expected_accuracy"] - r["expected_accuracy_by_agreement"]) > GUIDANCE_GAP
    assert "differ" in r["guidance"] and "conservative" in r["guidance"]
    assert "hand-typed" not in r["guidance"]


def test_route_below_floor_withholds_both(idx):
    r = idx.route("0.4")   # top 0.36 < 0.47
    assert r["expected_accuracy"] is None and r["expected_accuracy_by_agreement"] is None
    assert "abstention floor" in r["guidance"]
    assert r["queue"] == "Billing and Payments"   # the vote is still reported


def test_band_lookup(idx):
    assert [idx.expected_accuracy(s) for s in (0.85, 0.7, 0.1)] == [0.97, 0.67, 0.46]
    assert [idx.expected_accuracy_from_agreement(s) for s in (1.0, 2 / 3, 1 / 3)] == [0.92, 0.80, 0.63]


def test_index_and_database_must_agree(tmp_path):
    with pytest.raises(RuntimeError, match="absent from the database"):
        build(tmp_path, drop=("t4",))


def test_collapse_duplicates_rules():
    vectors = np.stack([unit(0.9, 1), unit(0.9, 1), unit(0.7, 2), unit(0.3, 3)])  # 0 and 1 identical
    order = np.array([0, 1, 2, 3])
    assert list(collapse_duplicates(order, vectors, keep=2)) == [0, 2]        # 1 collapsed into 0
    assert list(collapse_duplicates(order, vectors, keep=4)) == [0, 1, 2, 3]  # shape is kept: reject appended back
    assert list(collapse_duplicates(order[:2], vectors, keep=5)) == [0, 1]    # fewer inputs than keep
