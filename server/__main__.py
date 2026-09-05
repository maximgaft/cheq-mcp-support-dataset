"""MCP server over the customer-support ticket corpus.

Runs on stdio. The host (Claude Code / Codex) supplies the model; this server
holds no LLM and needs no API key. It answers two kinds of question - what is in
the queue, and how was something like this handled before - and is honest about
what the data cannot answer.

Every threshold and metric in a tool description is rendered from the files the
build and the eval wrote (data/interim/*.json) - the same files the server serves
from - so the text a model reads cannot drift from the numbers it applies. Before
`make build` the server still starts and the descriptions carry no figures;
before `make eval` the routing figures are absent and the vote falls back to k=1.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from server import db
from server.retrieval import GUIDANCE_GAP, Index

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
DB_PATH = Path(os.environ.get("CHEQ_DB", INTERIM / "08_tickets.duckdb"))
INDEX_PATH = INTERIM / "06_index.npz"
THRESHOLDS_PATH = INTERIM / "07_thresholds.json"
ROUTING_METRICS_PATH = INTERIM / "routing_metrics.json"
PROVENANCE_PATH = INTERIM / "08_provenance.json"

DEFAULT_K = 5
MAX_K = 20

FACET_COLUMNS = ["queue", "priority", "type", "language"]


def _json(path: Path) -> dict:
    """A build artifact, or {} before the stage that writes it has run."""
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _pct(value: float) -> str:
    return f"{100 * value:.0f}%"


_thresholds = _json(THRESHOLDS_PATH)          # written by pipeline/07_calibrate.py
_metrics = _json(ROUTING_METRICS_PATH)        # written by evals/run_routing.py
_provenance = _json(PROVENANCE_PATH)          # written by pipeline/08_database.py
_bands = sorted(_metrics.get("reliability_bands") or [], key=lambda b: -b["min_similarity"])
_by_language = _metrics.get("by_language") or {}

CANNOT_ANSWER = [
    "Anything over time. There are no timestamps - no trends, seasonality, "
    "volume spikes, backlog, time-to-first-response or time-to-resolution.",
    "Anything per customer. There is no customer identifier and personal details "
    "are replaced by placeholders like <name> and <acc_num>, so repeat contacters, "
    "per-account load and any link to churn are all unavailable.",
    "Agent or team performance. There is no assignee.",
    "Outcome quality. Nothing records whether a ticket was resolved well, reopened, "
    "or escalated, and there is no satisfaction score.",
    "Money. There is no cost, revenue or refund amount anywhere.",
]

SCHEMA_NOTES = [
    "ticket_tags holds one row per (ticket, tag). Joining it multiplies ticket "
    "rows, so count tickets with count(DISTINCT ticket_id), never count(*). Tag "
    "spellings are folded (case and separators normalised, each shown in its most "
    "common spelling) and cells that held a comma-joined list are split, so GROUP BY "
    "tag counts one tag once.",
    "There are deliberately no tag_1..tag_8 columns. The source stores tags "
    "positionally, so the same tag appears in different slots on different rows "
    "and GROUP BY tag_1 returns a confidently wrong answer. Use ticket_tags.",
    "split is train/val/test and is_indexable marks tickets usable as retrieval "
    "precedent: body and answer both at least 5 words, and the answer in the same "
    "language as the body. Only tickets with split = 'train' AND is_indexable are in "
    "the retrieval index"
    + (f" - {_provenance['tickets_indexed']:,} of them" if "tickets_indexed" in _provenance else "")
    + ". Non-indexable tickets are still real tickets and count in every aggregate; "
    "val and test exist so retrieval and routing can be measured on unseen data.",
    "language is detected from the ticket text (body first, answer if the body is "
    "too short to tell). language_label is what the source dataset said, and it is "
    "wrong on about 10% of rows - thousands of tickets labelled de whose text is "
    "English. Tickets in a third language were dropped. Filter and group by language; "
    "language_label exists only so the defect can be measured "
    "(WHERE language != language_label).",
    "The corpus is synthetic, generated for classifier training. It demonstrates "
    "mechanism; it is not evidence about real customers.",
    "reply_state says whether the customer could act on that first reply alone: "
    "resolved, actionable_ask (it named something fetchable, like the system logs "
    "or an invoice number), or dead_end (it asked for 'details', only acknowledged, "
    "or was not a support reply at all). It comes from a one-off LLM labelling pass "
    "and agrees with 40 hand-adjudicated tickets 75% of the time (reports/labels.md). "
    "Dead ends cost a round trip and buy nothing, and about half of all first replies "
    "are one. Prefer actionable_ask precedents when drafting.",
]


def _instructions() -> str:
    served = _provenance.get("tickets_served")
    return (
        "Customer-support ticket corpus: "
        + (f"{served:,} " if served else "")
        + "bilingual (en/de) tickets across 10 queues. Call get_schema first - it lists "
        "the columns, the values each categorical column can take, where the rows came "
        "from, and the questions this data cannot answer. Use run_sql for counts, mixes "
        "and distributions; find_similar_tickets and suggest_routing for a specific ticket."
    )


def _schema_description() -> str:
    return (
        "Describe the ticket database: tables, columns, the values each categorical column "
        "can take, where the rows came from, and what this data cannot answer.\n\n"
        "Call this before writing any SQL. The categorical value lists matter - queue and "
        "priority values are specific strings, and guessing them produces queries that run "
        "fine and return zero rows. `source_rows` reconciles the served ticket count with "
        "the raw download, so \"how many tickets are there\" has a checkable answer."
    )


def _find_description() -> str:
    head = (
        "Find past tickets similar to some text, with the replies that were sent.\n\n"
        "Pass the ticket's own text, subject and body together. Every hit carries "
        "reply_state - whether that past reply resolved the ticket, named something the "
        "customer could fetch (actionable_ask), or was a dead end that asked for \"details\" "
        "and cost a round trip. About half are dead ends; prefer the actionable ones when "
        "drafting a first reply.\n\n"
    )
    if _thresholds:
        off = _thresholds["off_topic_accepted"]
        off_text = ("rejected every off-topic query tested" if off == 0
                    else f"let through {_pct(off)} of the off-topic queries tested")
        floor = (
            f"has_precedent is false when the closest match is below a floor of "
            f"{_thresholds['similarity_floor']:.2f}, calibrated on real tickets: it accepts "
            f"{_pct(_thresholds['real_accepted'])} of them and {off_text}. For a real ticket, "
            "false means there is no usable precedent - route it to a human rather than "
            "drafting. For a question typed by a person the floor is advisory: hand-typed "
            "text scores lower than this corpus's tickets whatever its topic, so the floor "
            f"wrongly refuses about {_pct(1 - _thresholds['handwritten_on_topic_accepted'])} "
            "of legitimate typed questions (reports/calibration.md). The hits are returned "
            "either way.\n\n"
        )
    else:
        floor = (
            "has_precedent applies a similarity floor that pipeline/07_calibrate.py fits on "
            "real tickets; the figures appear here after `make build`. For a real ticket, "
            "false means there is no usable precedent - route it to a human rather than "
            "drafting. For a question typed by a person the floor is advisory. The hits are "
            "returned either way.\n\n"
        )
    indexed = (f"{_provenance['tickets_indexed']:,} indexed tickets"
               if "tickets_indexed" in _provenance else "the indexed tickets")
    tail = (
        f"Searches {indexed}. Similarity is on a centred scale where 0.0 means unrelated and "
        f"0.6-0.9 is genuinely similar. Each hit includes the full original answer, so k is "
        f"capped at {MAX_K} and defaults to {DEFAULT_K}."
    )
    return head + floor + tail


def _routing_description() -> str:
    head = (
        "Predict which queue and priority a ticket belongs to, with how much to trust the "
        "prediction and the neighbours it came from.\n\n"
        # k=1 is also what Index falls back to when the metrics file is absent.
        f"Nearest-neighbour vote over the indexed tickets, k={_metrics.get('k', 1)}: k was "
        "swept on the validation split, and a wider vote measurably degrades it by flooding "
        "the small queues with the majority class. "
    )
    if _metrics and _bands:
        top, low = _bands[0], _bands[-1]
        below = (f", below {_bands[-2]['min_similarity']:.2f} only {_pct(low['accuracy'])}"
                 if len(_bands) > 1 else "")
        figures = (
            f"Overall {_metrics['queue_macro_f1']:.3f} macro-F1 across 10 queues against a "
            f"{_metrics['baseline_macro_f1']:.3f} majority baseline, but "
            f"{_by_language.get('en', float('nan')):.3f} on English and "
            f"{_by_language.get('de', float('nan')):.3f} on German - weight German predictions "
            "accordingly.\n\n"
            f"Two confidence figures come back, both fitted on {_metrics['n_val']:,} validation "
            f"tickets and checked on {_metrics['n_test']:,} held-out test tickets "
            "(reports/routing.md). expected_accuracy comes from top_similarity: at or above "
            f"{top['min_similarity']:.2f} predictions were right {_pct(top['accuracy'])} of the "
            f"time{below}. expected_accuracy_by_agreement comes from how many of the "
            f"{_metrics.get('agreement_n', 3)} nearest neighbours share a queue; it is the "
            "weaker signal and its fitted values ran high on unseen tickets, so read it as a "
            "ceiling. "
        )
    else:
        figures = (
            "Its accuracy, per-language scores and the two confidence calibrations appear "
            "here after `make eval` (reports/routing.md).\n\n"
            "expected_accuracy comes from top_similarity; expected_accuracy_by_agreement from "
            "how many of the 3 nearest neighbours share a queue, the weaker signal - read it "
            "as a ceiling. "
        )
    tail = (
        "The two often differ on real tickets - when they differ by more than "
        f"{GUIDANCE_GAP:.2f} a guidance string says so, and the lower figure is the "
        "conservative read. Below the abstention floor both are withheld: route it to a "
        "human.\n\n"
        "The neighbours are returned so the prediction can be checked rather than taken on "
        "trust; `voted` marks the ones that decided it."
    )
    return head + figures + tail


mcp = MCPServer(name="cheq-tickets", instructions=_instructions())

_con = None
_index = None


def connection():
    global _con
    if _con is None:
        if not DB_PATH.exists():
            raise FileNotFoundError(
                f"{DB_PATH} not found. Run `make build` first (fetch + 8 pipeline stages, ~2 min)."
            )
        _con = db.connect(DB_PATH)
    return _con


def index() -> Index:
    """Loaded on first retrieval call - it pulls in the embedding model."""
    global _index
    if _index is None:
        for path in (INDEX_PATH, THRESHOLDS_PATH):
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found. Run `make build` first (fetch + 8 pipeline stages, ~2 min)."
                )
        _index = Index(INDEX_PATH, THRESHOLDS_PATH, connection(), ROUTING_METRICS_PATH)
    return _index


def _empty(text: str) -> dict | None:
    if not text or not text.strip():
        return {"error": "text is empty - pass the ticket's subject and body"}
    return None


@mcp.tool(description=_schema_description())
def get_schema() -> dict:
    """Tables, facets, provenance, notes, and the cannot_answer list."""
    con = connection()
    tables = {}
    for table in ("tickets", "ticket_tags"):
        columns = [
            {"name": name, "type": dtype}
            for name, dtype, *_ in con.execute(f"DESCRIBE {table}").fetchall()
        ]
        count = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        tables[table] = {"rows": count, "columns": columns}

    facets = {
        column: [
            {"value": value, "tickets": n}
            for value, n in con.execute(
                f"SELECT {column}, count(*) FROM tickets GROUP BY 1 ORDER BY 2 DESC"
            ).fetchall()
        ]
        for column in FACET_COLUMNS
    }
    top_tags = con.execute(
        "SELECT tag, count(*) AS n FROM ticket_tags GROUP BY 1 ORDER BY 2 DESC LIMIT 20"
    ).fetchall()
    distinct_tags = con.execute("SELECT count(DISTINCT tag) FROM ticket_tags").fetchone()[0]

    return {
        "tables": tables,
        "categorical_values": facets,
        "tags": {"distinct": distinct_tags, "most_common": [t for t, _ in top_tags]},
        "source_rows": _json(PROVENANCE_PATH) or None,
        "notes": SCHEMA_NOTES,
        "cannot_answer": CANNOT_ANSWER,
    }


@mcp.tool()
def run_sql(query: str, max_rows: int = db.DEFAULT_MAX_ROWS) -> dict:
    """Run one read-only SELECT against the ticket database and return the rows.

    Use this for counts, mixes, distributions and filtered lookups - "how many
    high-priority Billing tickets are in German", "which tags appear most in
    critical tickets", "how many queues have over 1,000 tickets".

    Write the SQL yourself from get_schema; there is no natural-language layer
    here. Aggregate rather than scanning: the result is capped at max_rows and a
    40,000-character budget, so SELECT * over many rows comes back truncated.
    Full ticket text is better obtained from the retrieval tools than from here.

    Only a single SELECT is accepted. The connection is read-only with file
    access disabled, so DDL, DML, COPY, ATTACH and chained statements are
    rejected rather than executed.
    """
    try:
        return db.run(connection(), query, max_rows=max_rows)
    except db.SqlRejected as exc:
        return {"error": str(exc), "query": query}


@mcp.tool(description=_find_description())
def find_similar_tickets(
    text: str,
    queue: str | None = None,
    language: str | None = None,
    k: int = DEFAULT_K,
) -> dict:
    """Nearest indexed tickets with their replies, and whether the closest is above the floor."""
    if error := _empty(text):
        return error
    idx = index()
    hits, top = idx.search(text, min(max(int(k), 1), MAX_K), queue=queue, language=language)
    if not hits:
        return {"error": "no tickets matched the given filters", "queue": queue, "language": language}

    result = {
        "floor": idx.floor,
        "top_similarity": top,
        "has_precedent": bool(top >= idx.floor),
        "results": hits,
    }
    if not result["has_precedent"]:
        result["guidance"] = (
            f"Closest match is {top:.3f}, below the {idx.floor} floor calibrated on real "
            "ticket text. For a real ticket this means there is no usable precedent - route "
            "it to a human instead of drafting a reply. For a question typed by a person the "
            "floor is advisory: typed text scores lower than this corpus's tickets whatever "
            "its topic, so read the hits with judgement."
        )
    return result


@mcp.tool(description=_routing_description())
def suggest_routing(text: str) -> dict:
    """Queue and priority by nearest-neighbour vote, with two calibrated confidence figures."""
    if error := _empty(text):
        return error
    return index().route(text)


if __name__ == "__main__":
    mcp.run()
