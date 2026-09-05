"""Read-only SQL access to the ticket database, and the guard around it.

Four independent layers keep a query from reading or changing anything outside
the two tables. None is sufficient alone, which is the point:

  read_only=True            blocks DDL and DML. The config alone does not -
                            CREATE TABLE still succeeds with file access off.
  enable_external_access    blocks all file and network I/O, so read_csv,
    =False                  read_parquet, glob, COPY..TO and ATTACH cannot
                            reach the disk.
  lock_configuration=True   blocks SET, so the layers above cannot be undone
                            from inside a query.
  exactly one SELECT        blocks semicolon chaining. DuckDB executes both
                            halves of "SELECT 1; SELECT 2" in a single call,
                            so checking that a query *starts* with SELECT is
                            not a check.

A denylist of dangerous function names would be the wrong shape. It enumerates
attacks, so it needs extending for every DuckDB function that learns to touch a
file, and it loses to name obfuscation. Disabling external access removes the
capability instead, which covers functions that do not exist yet.

Two more layers bound what a legitimate query can cost:

  a cursor per call         MCP runs synchronous tools in worker threads, so two
                            run_sql calls in one host turn execute concurrently.
                            A DuckDB connection is not safe to share that way -
                            in testing, two threads on one connection returned
                            each other's rows with no error raised. cursor()
                            opens a sibling connection that inherits read_only
                            and the locked config; each call gets its own and
                            closes it.
  memory_limit, threads,    memory_limit caps buffer-managed work (sorts, joins,
    10s watchdog            aggregates) and turns a runaway into an error the
                            caller sees; threads keeps one query from taking the
                            machine. Scalar allocations such as repeat() are not
                            tracked by memory_limit and are bounded only by the
                            watchdog, which interrupts the call's own cursor.

Known residual: metadata is readable. PRAGMA parses with statement type SELECT,
and duckdb_databases() / duckdb_settings() are plain SELECTs; they report the
database file's absolute path, and a rejected duckdb_extensions() call echoes the
extension directory in its error text. On a local stdio server that is the
caller's own home directory. It would need redacting before the HTTP deployment
the design note sketches. Documented rather than papered over.
"""

from __future__ import annotations

import threading
from pathlib import Path

import duckdb

HARDENED = {
    "enable_external_access": False,
    "lock_configuration": True,
    "memory_limit": "1GB",
    "threads": 2,
}

DEFAULT_MAX_ROWS = 100
ROW_CEILING = 500
MAX_TOTAL_CHARS = 40_000
TIMEOUT_SECONDS = 10
MARKER = " ... [cut to fit the response budget]"
MAX_COLUMN_NAME = 200   # a SELECT alias can be a megabyte; the budget is on cells, so bound names separately


class SqlRejected(Exception):
    """The query did not pass the guard, or could not be completed safely."""


def connect(path: Path | str) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(path), read_only=True, config=HARDENED)


def validate(sql: str) -> None:
    try:
        statements = duckdb.extract_statements(sql)
    except duckdb.Error as exc:
        raise SqlRejected(f"could not parse the query: {exc}") from exc

    if len(statements) != 1:
        raise SqlRejected(
            f"expected exactly one statement, got {len(statements)}. "
            "Send one SELECT per call - chained statements are not executed."
        )
    kind = statements[0].type.name
    if kind != "SELECT":
        raise SqlRejected(
            f"only SELECT is allowed, this is a {kind} statement. "
            "The database is read-only and has no file access."
        )


def _cell(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def fit_row(row: list, budget: int) -> str:
    """Shrink a lone over-budget row in place until its cells total at most `budget`
    characters, and say what was done.

    Long cells are cut first, largest first, each ending in MARKER. When the cells
    are too short for cutting to help - a wide row of small values - trailing cells
    are blanked instead. Every value is handled as text, so a numeric row cannot
    break the cut. The budget is a bound on cell characters, not on the JSON
    envelope around them."""
    sizes = [len(str(v)) for v in row]
    over = sum(sizes) - budget
    cut = 0
    for j in sorted(range(len(row)), key=lambda j: -sizes[j]):
        if over <= 0 or sizes[j] <= 4 * len(MARKER):   # keep a meaningful prefix, or leave it to blanking
            break
        text = str(row[j])
        take = min(over + len(MARKER), len(text))
        row[j] = text[: len(text) - take] + MARKER
        over -= sizes[j] - len(row[j])
        sizes[j] = len(row[j])
        cut += 1
    blanked = 0
    for j in range(len(row) - 1, -1, -1):
        if over <= 0:
            break
        over -= sizes[j]
        row[j], sizes[j] = "", 0
        blanked += 1
    parts = [f"{cut} long cell(s) cut" if cut else "", f"the last {blanked} of {len(row)} cells blanked" if blanked else ""]
    return " and ".join(part for part in parts if part)


def run(con: duckdb.DuckDBPyConnection, sql: str, max_rows: int = DEFAULT_MAX_ROWS) -> dict:
    """Validate, execute on a private cursor under a timeout, and return a
    size-bounded result."""
    validate(sql)
    max_rows = max(1, min(int(max_rows), ROW_CEILING))

    cur = con.cursor()
    watchdog = threading.Timer(TIMEOUT_SECONDS, cur.interrupt)
    watchdog.start()
    try:
        cur.execute(sql)
        columns = [d[0] if len(d[0]) <= MAX_COLUMN_NAME else d[0][:MAX_COLUMN_NAME] + MARKER
                   for d in cur.description]
        fetched = cur.fetchmany(max_rows + 1)  # one extra reveals "there is more"
    except duckdb.InterruptException as exc:
        raise SqlRejected(
            f"query ran longer than {TIMEOUT_SECONDS}s and was cancelled. "
            "Aggregate rather than scanning, or narrow the WHERE clause."
        ) from exc
    except duckdb.Error as exc:
        raise SqlRejected(str(exc)) from exc
    finally:
        watchdog.cancel()
        cur.close()

    rows: list[list] = []
    chars = 0
    for record in fetched[:max_rows]:
        row = [_cell(v) for v in record]
        size = sum(len(str(v)) for v in row)
        if chars + size > MAX_TOTAL_CHARS and rows:
            break
        rows.append(row)
        chars += size

    if chars > MAX_TOTAL_CHARS:  # only a lone first row can get here: shrink it to the budget
        what = fit_row(rows[0], MAX_TOTAL_CHARS)
        truncated = (
            f"returned one row of {chars:,} characters, over the {MAX_TOTAL_CHARS:,}-character "
            f"budget - {what}; select fewer or shorter columns"
        )
    elif len(rows) < len(fetched[:max_rows]):
        truncated = (
            f"stopped at {len(rows)} rows to stay under the {MAX_TOTAL_CHARS:,}-character "
            "response budget - select fewer columns, or aggregate"
        )
    elif len(fetched) > max_rows:
        truncated = f"more rows exist beyond max_rows={max_rows}"
    else:
        truncated = None

    return {"columns": columns, "rows": rows, "row_count": len(rows), "truncated": truncated}
