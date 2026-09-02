"""MCP server over the customer-support ticket corpus.

Runs on stdio. The host (Claude Code / Codex) supplies the model; this server
holds no LLM and needs no API key. It answers two kinds of question - what is in
the queue, and how was something like this handled before - and is honest about
what the data cannot answer.
"""

from __future__ import annotations

import os
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from server import db
from server.retrieval import Index, query_text

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
DB_PATH = Path(os.environ.get("CHEQ_DB", INTERIM / "08_tickets.duckdb"))
INDEX_PATH = INTERIM / "06_index.npz"
THRESHOLDS_PATH = INTERIM / "07_thresholds.json"
ROUTING_METRICS_PATH = INTERIM / "routing_metrics.json"

MAX_K = 20
COVERAGE_FACETS = ["queue", "priority", "type", "language"]

FACET_COLUMNS = ["queue", "priority", "type", "language", "split", "source"]

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
    "rows, so count tickets with count(DISTINCT ticket_id), never count(*).",
    "There are deliberately no tag_1..tag_8 columns. The source stores tags "
    "positionally, so the same tag appears in different slots on different rows "
    "and GROUP BY tag_1 returns a confidently wrong answer. Use ticket_tags.",
    "is_indexable marks tickets usable as retrieval precedent: they have an "
    "answer and a body of at least 5 words. Non-indexable tickets are still real "
    "tickets and still count in every aggregate.",
    "split is train/val/test. Only train tickets are in the retrieval index; "
    "val and test exist so retrieval and routing can be measured on unseen data.",
    "The corpus is synthetic, generated for classifier training. It demonstrates "
    "mechanism; it is not evidence about real customers.",
]

mcp = MCPServer(
    name="cheq-tickets",
    instructions=(
        "Customer-support ticket corpus: 40,064 bilingual (en/de) tickets across "
        "10 queues. Call get_schema first - it lists the columns, the values each "
        "categorical column can take, and the questions this data cannot answer. "
        "Use run_sql for counts, mixes and distributions."
    ),
)

_con = None
_index = None


def connection():
    global _con
    if _con is None:
        if not DB_PATH.exists():
            raise FileNotFoundError(
                f"{DB_PATH} not found. Build it with: uv run python pipeline/08_database.py"
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
                    f"{path} not found. Build it with: uv run python pipeline/06_embed.py "
                    "&& uv run python pipeline/07_calibrate.py"
                )
        _index = Index(INDEX_PATH, THRESHOLDS_PATH, connection(), ROUTING_METRICS_PATH)
    return _index


@mcp.tool()
def get_schema() -> dict:
    """Describe the ticket database: tables, columns, the values each categorical
    column can take, and what this data cannot answer.

    Call this before writing any SQL. The categorical value lists matter - queue
    and priority values are specific strings, and guessing them produces queries
    that run fine and return zero rows.
    """
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


@mcp.tool()
def find_similar_tickets(
    text: str,
    mode: str = "precedent",
    queue: str | None = None,
    language: str | None = None,
    k: int = 5,
) -> dict:
    """Find past tickets similar to some text, with their resolutions.

    mode="precedent" (default) is for drafting a reply. It applies a calibrated
    similarity floor and returns has_precedent=false when the closest matches are
    not close enough to draft from. When that happens, escalate to a human rather
    than writing a reply anyway - the matches returned alongside are for context,
    not for grounding.

    Pass the ticket's own text - subject and body together. The floor was
    calibrated on real tickets, which is the shape this receives in production,
    and it accepts 98% of them while rejecting every off-topic query tested.

    mode="explore" is the right mode for a question typed by a person, and the
    reason is measured. Text written by a human scores about 0.16 lower than text
    from this corpus's generator whatever its topic - which is about the same size
    as the gap between on-topic and off-topic. A single similarity threshold
    cannot separate those two effects, so on hand-typed questions the floor
    wrongly rejects roughly a third of legitimate ones. Measured means: real
    tickets 0.66, handwritten on-topic 0.50, handwritten off-topic 0.35. Details
    in reports/calibration.md.

    Both modes search the same 31,795 indexed tickets; only the floor differs.

    Every hit includes the full original answer, so k is capped at 20 and defaults
    to 5. Similarity is on a centred scale where 0.0 means unrelated - values
    around 0.6-0.9 are genuinely similar, and the floor sits at 0.45.
    """
    idx = index()
    if mode not in ("precedent", "explore"):
        return {"error": f"mode must be 'precedent' or 'explore', got {mode!r}"}

    hits, top = idx.search(text, min(max(int(k), 1), MAX_K), queue=queue, language=language)
    if not hits:
        return {"error": "no tickets matched the given filters", "queue": queue, "language": language}

    result = {"mode": mode, "floor": idx.floor, "top_similarity": top, "results": hits}
    if mode == "precedent":
        result["has_precedent"] = bool(top >= idx.floor)
        if not result["has_precedent"]:
            result["guidance"] = (
                f"Closest match is {top:.3f}, below the {idx.floor} floor. There is no usable "
                "precedent for this - route it to a human instead of drafting a reply."
            )
    return result


@mcp.tool()
def suggest_routing(text: str) -> dict:
    """Predict which queue and priority a ticket belongs to, with how much to
    trust the prediction, and the neighbours it came from.

    Read expected_accuracy before acting on the label. It is measured, not
    estimated: on 4,008 held-out tickets, predictions whose top_similarity was
    above 0.80 were right 98.5% of the time, and those below 0.60 were right 49%
    of the time. A low value means route it to a human.

    The neighbours are returned so the prediction can be checked rather than
    taken on trust; `voted` marks the ones that actually decided it. There is no
    k parameter because k was measured rather than chosen - widening the vote
    degrades it sharply, since a larger neighbourhood floods the small queues
    with the majority class.

    Overall: 0.697 macro-F1 across 10 queues against a 0.045 majority baseline,
    but 0.767 on English and 0.590 on German. Weight German predictions
    accordingly. Full breakdown in reports/routing.md.
    """
    return index().route(text)


@mcp.tool()
def precedent_coverage(group_by: str = "queue", per_group: int = 20) -> dict:
    """Measure where the corpus has usable precedent and where it does not.

    Samples held-out tickets, searches the index with each one, and reports the
    share that find a precedent above the floor. Low coverage in a group means
    agents handling that kind of ticket have little institutional precedent to
    work from - which is a documentation gap, not a tool failure.

    This is the one question the corpus can answer that no single retrieval call
    can: it is a property of the whole collection rather than of one result.
    Queries are drawn from val/test, so they are never in the index.

    Slower than the other tools - it embeds group_by count x per_group tickets,
    a few seconds. group_by must be queue, priority, type or language.
    """
    if group_by not in COVERAGE_FACETS:
        return {"error": f"group_by must be one of {COVERAGE_FACETS}, got {group_by!r}"}

    per_group = min(max(int(per_group), 5), 50)
    rows = connection().execute(f"""
        SELECT ticket_id, grp, subject, body FROM (
            SELECT ticket_id, {group_by} AS grp, subject, body,
                   row_number() OVER (PARTITION BY {group_by} ORDER BY hash(ticket_id)) AS rn
            FROM tickets WHERE split IN ('val', 'test')
        ) WHERE rn <= {per_group}
    """).df()

    idx = index()
    rows["top1"] = [
        idx.top1(query_text(s, b)) for s, b in zip(rows.subject, rows.body)
    ]

    corpus_median = float(rows.top1.median())
    groups = []
    for name, g in rows.groupby("grp"):
        groups.append({
            group_by: name,
            "sampled": len(g),
            "median_similarity": round(float(g.top1.median()), 3),
            "p25_similarity": round(float(g.top1.quantile(0.25)), 3),
            "share_below_corpus_median": round(float((g.top1 < corpus_median).mean()), 3),
        })
    groups.sort(key=lambda row: row["median_similarity"])

    return {
        "group_by": group_by,
        "corpus_median_similarity": round(corpus_median, 3),
        "abstention_floor": idx.floor,
        "sampled_from": "val + test (never in the index)",
        "groups": groups,
        "note": (
            "Thinnest precedent first. These are similarity distributions, not pass "
            "rates against the abstention floor - almost every in-domain ticket clears "
            f"that floor ({idx.floor}), so a share against it carries no information. "
            "Read the medians comparatively: a group well below "
            f"{round(corpus_median, 3)} is one where agents have weaker precedent to "
            "work from, which points at a documentation gap rather than a tool failure."
        ),
    }


if __name__ == "__main__":
    mcp.run()
