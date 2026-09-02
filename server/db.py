"""Read-only SQL access to the ticket database, and the guard around it.

Four independent layers. None is sufficient alone, which is the point:

  read_only=True            blocks DDL and DML. The config alone does not -
                            CREATE TABLE still succeeds with file access off.
  enable_external_access    blocks all file and network I/O, so read_csv,
    =False                  read_parquet, glob, COPY..TO and ATTACH cannot
                            reach the disk.
  lock_configuration=True   blocks SET, so the layer above cannot be undone
                            from inside a query.
  exactly one SELECT        blocks semicolon chaining. DuckDB executes both
                            halves of "SELECT 1; SELECT 2" in a single call,
                            so checking that a query *starts* with SELECT is
                            not a check.

A denylist of dangerous function names would be the wrong shape. It enumerates
attacks, so it needs extending for every DuckDB function that learns to touch a
file, and it loses to name obfuscation. Disabling external access removes the
capability instead, which covers functions that do not exist yet.

Known residual: PRAGMA parses with statement type SELECT and therefore passes
the statement check. Under a read-only connection with a locked configuration it
can only report metadata - it returned the database path and build version in
testing. Documented rather than papered over.
"""

from __future__ import annotations

import threading
from pathlib import Path

import duckdb

HARDENED = {"enable_external_access": False, "lock_configuration": True}

DEFAULT_MAX_ROWS = 100
ROW_CEILING = 500
MAX_TOTAL_CHARS = 40_000
TIMEOUT_SECONDS = 10


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


def run(con: duckdb.DuckDBPyConnection, sql: str, max_rows: int = DEFAULT_MAX_ROWS) -> dict:
    """Validate, execute under a timeout, and return a size-bounded result."""
    validate(sql)
    max_rows = max(1, min(int(max_rows), ROW_CEILING))

    watchdog = threading.Timer(TIMEOUT_SECONDS, con.interrupt)
    watchdog.start()
    try:
        cursor = con.execute(sql)
        columns = [d[0] for d in cursor.description]
        fetched = cursor.fetchmany(max_rows + 1)  # one extra reveals "there is more"
    except duckdb.InterruptException as exc:
        raise SqlRejected(
            f"query ran longer than {TIMEOUT_SECONDS}s and was cancelled. "
            "Aggregate rather than scanning, or narrow the WHERE clause."
        ) from exc
    except duckdb.Error as exc:
        raise SqlRejected(str(exc)) from exc
    finally:
        watchdog.cancel()

    rows: list[list] = []
    chars = 0
    for record in fetched[:max_rows]:
        row = [_cell(v) for v in record]
        chars += sum(len(str(v)) for v in row)
        if chars > MAX_TOTAL_CHARS and rows:
            break
        rows.append(row)

    if len(rows) < len(fetched[:max_rows]):
        truncated = (
            f"stopped at {len(rows)} rows to stay under the {MAX_TOTAL_CHARS:,}-character "
            "response budget - select fewer columns, or aggregate"
        )
    elif len(fetched) > max_rows:
        truncated = f"more rows exist beyond max_rows={max_rows}"
    else:
        truncated = None

    return {"columns": columns, "rows": rows, "row_count": len(rows), "truncated": truncated}
