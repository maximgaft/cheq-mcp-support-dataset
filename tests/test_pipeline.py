"""Pure functions from the pipeline and the eval, checked on hand-sized inputs."""

import hashlib
import importlib

import certifi
import numpy as np
import pandas as pd
import pytest

from pipeline.label import collapse

filter03 = importlib.import_module("pipeline.03_filter")
fetch00 = importlib.import_module("pipeline.00_fetch")
db08 = importlib.import_module("pipeline.08_database")
routing = importlib.import_module("evals.run_routing")


@pytest.mark.parametrize("text, lang", [
    ("The printer is not working and we need help with the driver for our office.", "en"),
    ("Der Drucker funktioniert nicht und wir brauchen Hilfe mit dem Treiber für das Büro.", "de"),
    ("Le serveur ne répond pas et nous avons besoin de votre aide pour la configuration.", "other"),
    ("Requesting assistance", None),   # no function words at all
    ("the der", None),                 # a tie decides nothing
    ("le la", None),                   # two foreign hits are below MIN_FOREIGN_HITS
])
def test_detect_language(text, lang):
    assert filter03.detect_language(text) == lang


def test_role_confused_catches_agent_text_in_the_body():
    frame = pd.DataFrame({
        "body": ["Dear <name>, thank you for contacting us about the invoice.",
                 "My printer stopped working after the update.",
                 "The projector shows no signal from the HDMI input since the firmware update"],
        "answer": ["irrelevant", "Please send the logs.",
                   "The projector shows no signal from the HDMI input since the firmware update"],
    })
    assert list(filter03.role_confused(frame)) == [True, False, True]


@pytest.mark.parametrize("kind, specific, state", [
    ("resolution", None, "resolved"),
    ("information_request", True, "actionable_ask"),
    ("information_request", False, "dead_end"),
    ("acknowledgement", None, "dead_end"),
    ("not_a_support_reply", None, "dead_end"),
])
def test_collapse(kind, specific, state):
    assert collapse(kind, specific) == state


def test_sha256_streams_the_whole_file(tmp_path):
    path = tmp_path / "blob"
    data = bytes(range(256)) * 10_000   # 2.5 MB, crosses the 1 MB chunk boundary
    path.write_bytes(data)
    assert fetch00.sha256(path) == hashlib.sha256(data).hexdigest()


def test_ca_bundle_honours_ssl_cert_file_then_falls_back_to_certifi(monkeypatch, tmp_path):
    corporate = tmp_path / "corporate-roots.pem"
    corporate.write_text("")
    monkeypatch.setenv("SSL_CERT_FILE", str(corporate))
    assert fetch00.ca_bundle() == str(corporate)
    monkeypatch.setenv("SSL_CERT_FILE", "")          # an empty value is not a bundle
    assert fetch00.ca_bundle() == certifi.where()
    monkeypatch.delenv("SSL_CERT_FILE")
    assert fetch00.ca_bundle() == certifi.where()


def test_canonical_spelling_is_the_most_common_then_alphabetical():
    assert db08.canonical(pd.Series(["API", "Api", "API"])) == "API"
    assert db08.canonical(pd.Series(["Api", "API"])) == "API"


def test_auc_is_tie_averaged():
    assert routing.auc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([True, True, False, False])) == 1.0
    assert routing.auc(np.array([0.9, 0.1]), np.array([False, True])) == 0.0
    assert routing.auc(np.array([0.5, 0.5, 0.5, 0.5]), np.array([True, False, True, False])) == 0.5


def test_jaccard_ignores_short_words():
    assert routing.jaccard("printer driver update", "printer driver update") == 1.0
    assert routing.jaccard("printer driver", "invoice refund") == 0.0
    assert routing.jaccard("the cat sat", "the dog sat") == 0.0   # no word has four letters


def test_band_table_and_labels():
    signal = np.array([0.85, 0.75, 0.65, 0.1])
    correct = np.array([True, True, False, False])
    table = routing.band_table(signal, correct, routing.SIM_BANDS)
    assert [(b["min"], b["n"], b["accuracy"]) for b in table] == [
        (0.8, 1, 1.0), (0.7, 1, 1.0), (0.6, 1, 0.0), (0.0, 1, 0.0)]
    assert [routing.sim_label(e, routing.SIM_BANDS) for e in routing.SIM_BANDS] == [
        ">= 0.80", "0.70 - 0.80", "0.60 - 0.70", "< 0.60"]
    assert [routing.agree_label(e) for e in routing.AGREE_BANDS] == [
        "3 of 3 agree", "2 of 3 agree", "1 of 3 agree"]


def test_scores_macro_f1_by_hand():
    out = routing.scores(np.array(["A", "A", "B"]), np.array(["A", "B", "B"]))
    assert out["accuracy"] == pytest.approx(2 / 3)
    assert out["macro_f1"] == pytest.approx(2 / 3)   # both classes: precision/recall 1 and 0.5


def test_vote_and_agreement():
    labels = np.array(["A", "B", "B"])
    positions = np.array([[0, 1, 2]])
    sims = np.array([[0.5, 0.4, 0.3]])
    assert routing.vote(labels, positions, sims, k=1)[0] == "A"
    assert routing.vote(labels, positions, sims, k=3)[0] == "B"    # 0.7 beats 0.5
    assert routing.agreement(labels, positions, n=3)[0] == pytest.approx(1 / 3)


def test_holdout_abstention_set_is_disjoint_and_scored_after_the_floor():
    import re

    import yaml
    fitting = yaml.safe_load(open("evals/abstention_queries.yaml"))
    holdout = yaml.safe_load(open("evals/abstention_holdout.yaml"))
    assert not {c["query"] for c in fitting} & {c["query"] for c in holdout}
    assert sum(c["expect"] == "no_precedent" for c in holdout) == 25
    assert sum(c["expect"] == "precedent" for c in holdout) == 15
    source = open("pipeline/07_calibrate.py").read()
    assert source.index("floor = ") < source.index("HOLDOUT.read_text()")   # never used to set the floor
    assert re.search(r"HOLDOUT\b", source[: source.index("floor = ")]) is None or "HOLDOUT = ROOT" in source[: source.index("floor = ")]
