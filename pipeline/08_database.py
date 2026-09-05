"""Stage 08 - build the DuckDB file the server queries, with a curated schema.

The schema is the guardrail. Two decisions matter more than the SQL:

  tag_1..tag_8 are not exposed. Tags are stored positionally in the source, so
  the same tag lands in different columns on different rows and `GROUP BY tag_1`
  returns a confidently wrong answer. Rather than warn a model off that query,
  the columns are absent and `ticket_tags` holds one row per assignment. The
  wrong query is unwritable, not discouraged.

  Pipeline bookkeeping is not served. `version` (a source-batch marker, 54%
  null), `source` (which CSV a row came from), `body_words`, and the raw answer
  labels `answer_kind` / `ask_is_specific` are dropped. Every column in the
  schema is a column a model may group by, so a column with no analytical
  meaning is a wrong answer waiting to be written. `split` and `is_indexable`
  stay: they let a user reconcile the indexed-ticket count from inside the tool.

  Where the rows came from is written to 08_provenance.json instead, so
  get_schema can reconcile the served count with the raw download.

Tables are materialised, not views over read_parquet: the serving connection
runs with file access disabled, so a view that reads a file would fail there.

The build connection writes, so it cannot be read-only. It does not need file
access either, because pandas reads the parquet and DuckDB ingests from memory.
"""

import importlib
import json
import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.label import collapse  # noqa: E402

SOURCES = importlib.import_module("pipeline.01_load").SOURCES

DATA = Path(__file__).resolve().parents[1] / "data"
LOADED = DATA / "interim" / "01_loaded.parquet"
FILTERED = DATA / "interim" / "03_corpus.parquet"
IN = DATA / "interim" / "05_split.parquet"
OUT = DATA / "interim" / "08_tickets.duckdb"
PROVENANCE = DATA / "interim" / "08_provenance.json"
LABELS = DATA / "answer_labels.parquet"

COLUMNS = [
    "ticket_id", "split", "language", "language_label", "queue", "priority",
    "type", "subject", "body", "answer", "is_indexable",
]
# answer_kind and ask_is_specific come from pipeline/label.py, which needs an API
# key. They are read only to derive reply_state - the collapsed label the business
# claim uses (the four-way label agrees with hand adjudication 70%, the collapse
# 75%; reports/labels.md) - and are not served. Derived here so no consumer
# re-implements the rule and gets it subtly wrong. Without the labels file,
# reply_state is null and the schema keeps its shape.
LABEL_COLUMNS = ["answer_kind", "ask_is_specific"]
TAG_SLOTS = [f"tag_{i}" for i in range(1, 9)]


def canonical(spellings: pd.Series) -> str:
    """The most common spelling in a group of variants; ties go alphabetically."""
    counts = spellings.value_counts()
    return sorted(counts[counts == counts.max()].index)[0]


def main() -> None:
    corpus = pd.read_parquet(IN)

    tickets = corpus[COLUMNS].copy()
    if LABELS.exists():
        labels = pd.read_parquet(LABELS).rename(columns={"kind": "answer_kind"})
        labelled = tickets[["ticket_id"]].merge(
            labels[["ticket_id", *LABEL_COLUMNS]], on="ticket_id", how="left")
        print(f"  joined {labels.ticket_id.nunique():,} answer labels")
    else:
        labelled = pd.DataFrame({"ticket_id": tickets.ticket_id, "answer_kind": None,
                                 "ask_is_specific": None})
        print("  no answer labels found - reply_state will be null; `make label-all` populates it")
    tickets["reply_state"] = [
        collapse(k, bool(sp)) if isinstance(k, str) else None
        for k, sp in zip(labelled.answer_kind, labelled.ask_is_specific)
    ]

    # Where the rows came from, so get_schema can reconcile 40k served tickets
    # with 61k downloaded rows instead of leaving a grader to wonder.
    loaded = pd.read_parquet(LOADED, columns=["source"])
    filtered = pd.read_parquet(FILTERED, columns=["body"])
    per_file = loaded.source.value_counts()
    excluded = int(per_file.get("german_norm", 0))
    dropped = int(len(loaded) - excluded - len(filtered))
    dup_rows = int(len(filtered) - len(corpus))
    dup_groups = int((filtered.body.value_counts() > 1).sum())
    assert len(loaded) - excluded - dropped - dup_rows == len(corpus)
    provenance = {
        "source_files": [{"file": SOURCES[src], "rows": int(n)} for src, n in per_file.items()],
        "rows_downloaded": int(len(loaded)),
        "excluded_file": {
            "file": SOURCES["german_norm"], "rows": excluded,
            "why": "a different dataset that shares its five column names (subject, body, "
                   "queue, priority, language) with the multi-language files and nothing else: "
                   "no answers, a 42-value queue taxonomy instead of 10, a 5-value priority "
                   "scale instead of 3, all German. It can join neither the aggregates nor "
                   "the precedent index.",
        },
        "dropped_invalid": {
            "rows": dropped,
            "why": "empty body, a body holding the agent's reply instead of the customer's "
                   "message, or a third language inside a two-language corpus",
        },
        "duplicates_collapsed": {
            "rows": dup_rows, "groups": dup_groups,
            "why": "the same ticket shipped in both multi-language files; kept once, and "
                   "before the train/val/test split so no test ticket has a twin in training",
        },
        "tickets_served": int(len(corpus)),
        "tickets_indexed": int((corpus.split.eq("train") & corpus.is_indexable).sum()),
        "generated_by": "pipeline/08_database.py",
    }
    PROVENANCE.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"  provenance: {len(loaded):,} downloaded - {excluded:,} other dataset - "
          f"{dropped:,} invalid - {dup_rows:,} duplicates = {len(corpus):,} served")
    tags = (
        corpus.melt(id_vars="ticket_id", value_vars=TAG_SLOTS, value_name="tag")[["ticket_id", "tag"]]
        .dropna(subset=["tag"])
        .assign(tag=lambda d: d.tag.str.split(","))  # 62 cells hold a comma-joined list of tags
        .explode("tag")
        .assign(tag=lambda d: d.tag.str.strip())
        .query("tag != ''")
        .reset_index(drop=True)
    )
    # Fold spelling variants onto one key - "access control" / "accesscontrol" /
    # "access-control" / "access_control", "API" / "Api" - and show each group as its
    # most common original spelling. Otherwise GROUP BY tag splits one tag across
    # its spellings: 173 separator groups and 27 case groups in the source.
    spellings = tags.tag.nunique()
    key = tags.tag.str.lower().str.replace(r"[\s\-_]+", "", regex=True)
    tags["tag"] = key.map(tags.tag.groupby(key).agg(canonical))
    tags = tags.drop_duplicates().sort_values(["ticket_id", "tag"])  # same tag twice on one ticket
    print(f"  tags: {spellings:,} distinct spellings -> {tags.tag.nunique():,} tags after folding")

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
