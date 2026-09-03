"""Stage 08 - build the DuckDB file the server queries, with a curated schema.

The schema is the guardrail. Two decisions matter more than the SQL:

  tag_1..tag_8 are not exposed. Tags are stored positionally in the source, so
  the same tag lands in different columns on different rows and `GROUP BY tag_1`
  returns a confidently wrong answer. Rather than warn a model off that query,
  the columns are absent and `ticket_tags` holds one row per assignment. The
  wrong query is unwritable, not discouraged.

  `version` is dropped. It is a source-batch marker with no meaning - 54% null,
  fully explained by `source` - and leaving it in only invites grouping by it.

Tables are materialised, not views over read_parquet: the serving connection
runs with file access disabled, so a view that reads a file would fail there.

The build connection writes, so it cannot be read-only. It does not need file
access either, because pandas reads the parquet and DuckDB ingests from memory.
"""

import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from label import collapse  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
IN = DATA / "interim" / "05_split.parquet"
OUT = DATA / "interim" / "08_tickets.duckdb"
LABELS = DATA / "answer_labels.parquet"

COLUMNS = [
    "ticket_id", "source", "split", "language", "queue", "priority", "type",
    "subject", "body", "answer", "body_words", "is_indexable",
]
# answer_kind and ask_is_specific come from pipeline/label.py, which needs an API
# key. The columns are created either way so the schema does not change shape
# depending on whether that pass has been run - they are simply null without it.
LABEL_COLUMNS = ["answer_kind", "ask_is_specific"]
# reply_state collapses those two into the distinction the business claim uses.
# Derived here so no consumer re-implements the rule and gets it subtly wrong.
TAG_SLOTS = [f"tag_{i}" for i in range(1, 9)]


def main() -> None:
    corpus = pd.read_parquet(IN)

    tickets = corpus[COLUMNS].copy()
    if LABELS.exists():
        labels = pd.read_parquet(LABELS).rename(columns={"kind": "answer_kind"})
        tickets = tickets.merge(labels[["ticket_id", *LABEL_COLUMNS]], on="ticket_id", how="left")
        print(f"  joined {labels.ticket_id.nunique():,} answer labels")
    else:
        for column in LABEL_COLUMNS:
            tickets[column] = None
        print("  no answer labels found - run `make label` to populate them")
    tickets["reply_state"] = [
        collapse(k, bool(sp)) if isinstance(k, str) else None
        for k, sp in zip(tickets.answer_kind, tickets.ask_is_specific)
    ]
    tags = (
        corpus.melt(id_vars="ticket_id", value_vars=TAG_SLOTS, value_name="tag")[["ticket_id", "tag"]]
        .dropna(subset=["tag"])
        .query("tag.str.strip() != ''")
        .drop_duplicates()  # 68 rows repeat a tag across slots; one assignment each
        .sort_values(["ticket_id", "tag"])
    )

    OUT.unlink(missing_ok=True)
    con = duckdb.connect(str(OUT))
    con.register("tickets_df", tickets)
    con.register("tags_df", tags)
    con.execute("CREATE TABLE tickets AS SELECT * FROM tickets_df")
    con.execute("CREATE TABLE ticket_tags AS SELECT * FROM tags_df")
    con.execute("CREATE INDEX idx_tags_ticket ON ticket_tags(ticket_id)")
    con.close()

    print(f"  tickets      {len(tickets):>7,} rows, {len(tickets.columns)} columns")
    print(f"  ticket_tags  {len(tags):>7,} rows, {tags.tag.nunique():,} distinct tags")

    # Verify the serving contract against what we just built, not in the abstract.
    srv = duckdb.connect(str(OUT), read_only=True,
                         config={"enable_external_access": False, "lock_configuration": True})

    cols = [r[0] for r in srv.execute("DESCRIBE tickets").fetchall()]
    leaked = [c for c in cols if c.startswith("tag_")]
    assert not leaked, f"positional tag columns leaked into the schema: {leaked}"
    assert srv.execute("SELECT count(*) FROM tickets").fetchone()[0] == len(tickets)
    assert srv.execute("SELECT count(*) FROM ticket_tags").fetchone()[0] == len(tags)

    print("\n  serving connection (read-only, no file access):")
    print(f"    tickets columns: {', '.join(cols)}")
    for label, sql in [
        ("read a local file", f"SELECT * FROM read_parquet('{IN}') LIMIT 1"),
        ("write a table", "CREATE TABLE z(a INT)"),
        ("unlock the config", "SET enable_external_access=true"),
    ]:
        try:
            srv.execute(sql)
            raise AssertionError(f"serving connection ALLOWED {label} - guard is broken")
        except duckdb.Error:
            print(f"    blocked: {label}")

    # count(DISTINCT ticket_id), not count(*): joining ticket_tags fans out one
    # row per tag, so count(*) here measures ticket-tag pairs. Removing the
    # positional-tag trap traded it for this one, which get_schema must state.
    top = srv.execute("""
        SELECT t.queue,
               count(DISTINCT t.ticket_id) AS tickets,
               count(DISTINCT tg.tag)      AS distinct_tags
        FROM tickets t LEFT JOIN ticket_tags tg USING (ticket_id)
        GROUP BY 1 ORDER BY 2 DESC LIMIT 3
    """).fetchall()
    print(f"    sample query: {top}")
    srv.close()
    print(f"\n  wrote {OUT.relative_to(DATA.parent)} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
