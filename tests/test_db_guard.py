"""The SQL guard, one layer per test, so a regression says which layer broke.

A file-backed database is required: DuckDB refuses read_only on :memory:.
"""

from concurrent.futures import ThreadPoolExecutor

import duckdb
import pytest

from server import db


@pytest.fixture
def con(tmp_path):
    path = tmp_path / "t.duckdb"
    writer = duckdb.connect(str(path))
    writer.execute("CREATE TABLE t AS SELECT range AS id, repeat('x', 10) AS s FROM range(1000)")
    writer.close()
    connection = db.connect(path)
    yield connection
    connection.close()


@pytest.fixture
def csv(tmp_path):
    path = tmp_path / "scratch.csv"
    path.write_text("a,b\n1,2\n")
    return path


def test_layer4_exactly_one_statement(con):
    with pytest.raises(db.SqlRejected, match="exactly one statement"):
        db.run(con, "SELECT 1; SELECT 2")
    with pytest.raises(db.SqlRejected, match="exactly one statement"):
        db.run(con, "SELECT 1 /* hidden */ ; DROP TABLE t")


@pytest.mark.parametrize("sql", [
    "CREATE TABLE z(a INT)",
    "INSERT INTO t VALUES (1, 'y')",
    "DELETE FROM t",
    "COPY t TO 'out.csv'",
    "ATTACH 'other.duckdb'",
    "SET enable_external_access = true",
    "INSTALL httpfs",
    "LOAD httpfs",
    "EXPORT DATABASE 'x'",
    "CALL duckdb_settings()",
    "EXPLAIN SELECT 1",
])
def test_layer4_only_select_is_accepted(con, sql):
    with pytest.raises(db.SqlRejected, match="only SELECT"):
        db.run(con, sql)


@pytest.mark.parametrize("sql", [
    "DESCRIBE t",
    "SHOW TABLES",
    "SUMMARIZE t",
    "WITH c AS (SELECT 1 AS a) SELECT a FROM c",
    "FROM t SELECT count(*)",
])
def test_select_shaped_conveniences_are_accepted(con, sql):
    # DuckDB types these as SELECT, so a host that writes DESCRIBE after get_schema is served.
    assert db.run(con, sql)["row_count"] >= 1


def test_parse_error_is_rejected_cleanly(con):
    with pytest.raises(db.SqlRejected, match="could not parse"):
        db.run(con, "SELEC 1")


def test_layer2_file_read_is_a_select_that_still_fails(con, csv):
    # Layer 4 lets it through - it IS a SELECT - which is why layer 2 exists.
    sql = f"SELECT * FROM read_csv('{csv}')"
    db.validate(sql)
    with pytest.raises(db.SqlRejected, match="(?i)disabled|permission"):
        db.run(con, sql)
    with pytest.raises(db.SqlRejected, match="(?i)disabled|permission"):
        db.run(con, f"SELECT * FROM '{csv}'")


def test_layers1_and_3_hold_on_a_bare_cursor(con):
    # Bypass validate() on purpose: the connection itself must refuse these.
    cur = con.cursor()
    for sql in ("CREATE TABLE z(a INT)",
                "SET enable_external_access = true",
                "SET memory_limit = '100GB'"):
        with pytest.raises(duckdb.Error):
            cur.execute(sql)
    cur.close()


def test_pragma_is_the_documented_residual(con):
    assert db.run(con, "PRAGMA version")["row_count"] == 1


def test_limits_are_applied_and_locked(con):
    rows = dict(db.run(con, "SELECT name, value FROM duckdb_settings() WHERE name IN "
                            "('access_mode', 'lock_configuration', 'enable_external_access', "
                            "'threads', 'memory_limit')")["rows"])
    assert rows == {"access_mode": "read_only", "lock_configuration": "true",
                    "enable_external_access": "false", "threads": "2",
                    "memory_limit": "953.6 MiB"}   # 1GB, printed by DuckDB in binary units


def test_row_cap_and_more_rows_flag(con):
    out = db.run(con, "SELECT id FROM t ORDER BY id", max_rows=10)
    assert out["row_count"] == 10 and "more rows" in out["truncated"]
    out = db.run(con, "SELECT id FROM t ORDER BY id", max_rows=10_000)
    assert out["row_count"] == db.ROW_CEILING


def test_character_budget_stops_early(con):
    out = db.run(con, "SELECT repeat('y', 15000) FROM range(5)")
    assert out["row_count"] < 5 and "character" in out["truncated"]


@pytest.mark.parametrize("sql", [
    "SELECT repeat('y', 1000000)",                                    # one huge cell
    "SELECT repeat('a', 50000) AS a, repeat('b', 50000) AS b",         # two large cells: both must shrink
    "SELECT " + ", ".join(f"{i} AS c{i}" for i in range(20000)),       # wide numeric row: cutting cannot help
])
def test_lone_row_budget_is_a_real_bound(con, sql):
    out = db.run(con, sql)
    assert out["row_count"] == 1
    assert sum(len(str(v)) for v in out["rows"][0]) <= db.MAX_TOTAL_CHARS
    assert "over the" in out["truncated"]


def test_medium_cells_keep_a_prefix_or_are_blanked_not_hollowed(con):
    row = db.run(con, "SELECT " + ", ".join(f"repeat('q', 75) AS c{i}" for i in range(1000)))["rows"][0]
    assert sum(len(str(v)) for v in row) <= db.MAX_TOTAL_CHARS
    assert all(cell == "" or not str(cell).startswith(db.MARKER) for cell in row)   # no cell is only a marker


def test_long_column_names_are_bounded(con):
    out = db.run(con, 'SELECT 1 AS "' + "a" * 300000 + '"')
    assert len(out["columns"][0]) <= db.MAX_COLUMN_NAME + len(db.MARKER)


def test_two_large_cells_are_both_cut(con):
    row = db.run(con, "SELECT repeat('a', 50000) AS a, repeat('b', 50000) AS b")["rows"][0]
    assert all(str(cell).endswith("budget]") for cell in row)


def test_watchdog_cancels_and_connection_survives(con, monkeypatch):
    monkeypatch.setattr(db, "TIMEOUT_SECONDS", 0.5)
    with pytest.raises(db.SqlRejected, match="cancelled"):
        db.run(con, "SELECT count(*) FROM range(200000) a, range(200000) b WHERE a.range < b.range")
    assert db.run(con, "SELECT count(*) FROM t")["rows"] == [[1000]]


def test_concurrent_calls_get_their_own_rows(con):
    def worker(limit: int) -> int:
        return sum(db.run(con, f"SELECT id FROM t ORDER BY id LIMIT {limit}")["row_count"] != limit
                   for _ in range(20))
    with ThreadPoolExecutor(4) as pool:
        assert list(pool.map(worker, [2, 3, 5, 7])) == [0, 0, 0, 0]
