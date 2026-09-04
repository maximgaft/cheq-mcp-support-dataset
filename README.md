# Support-ticket MCP server

An MCP server that answers natural-language questions about 40,047 bilingual
(English/German) customer-support tickets — and refuses the ones the data cannot
answer.

It exposes four tools to Claude Code or Codex: describe the data, query it with
SQL, find how similar tickets were handled before, and predict where a ticket
should be routed with a measured confidence figure attached.

**There is no API key.** The server holds no LLM — your MCP client supplies the
model, and the server supplies grounded facts and refusals. Everything below runs
offline.

The reasoning behind the design is one page: **[docs/design.pdf](docs/design.pdf)**.

[![design note](docs/design_preview.png)](docs/design.pdf)

---

## Quick start

```bash
git clone <this repo> && cd cheq
uv sync          # installs deps (~1 GB: torch)
make build       # fetch data + build the index (~2 min first run, ~80s after)
make eval        # routing + label agreement (~30s)
```

The build is deterministic: a fresh clone reproduces the published numbers
exactly — 0.707 macro-F1, 0.782 English / 0.542 German, abstention floor 0.47.

Then point your MCP client at it — see below — and ask it something.

> **First run downloads ~500 MB of embedding-model weights** from Hugging Face
> into `~/.cache/huggingface`. It looks like a hang; it isn't. Subsequent runs
> are instant.

## Requirements

- **Python 3.14** and [uv](https://docs.astral.sh/uv/)
- ~2 GB free disk (1 GB packages, 0.5 GB model weights, 0.15 GB build artifacts)
- No API key, no network at query time, no database server, no vector database

Dependencies are pinned in `uv.lock`: `duckdb`, `mcp`, `numpy`, `pandas`,
`pyarrow`, `pyyaml`, `sentence-transformers`, `torch`, `ruff`, and `anthropic` for the
optional labelling pass.

## Environment variables

None are required. One is optional:

| variable | default | purpose |
|---|---|---|
| `CHEQ_DB` | `data/interim/08_tickets.duckdb` | point the server at a different database file |
| `ANTHROPIC_API_KEY` | — | **only** for `make label`, the one-off pass that classifies what each first reply did. Its output is committed, so `make build` and `make eval` never need it. |

## Connect it to Claude Code

The repo ships a project-scoped [`.mcp.json`](.mcp.json), so **opening this
directory in Claude Code is enough** — the server is picked up automatically on
the next session.

To register it globally instead:

```bash
claude mcp add cheq-tickets -- uv run --directory /absolute/path/to/cheq python -m server
```

## Connect it to Codex

Add this to `~/.codex/config.toml`:

```toml
[mcp_servers.cheq-tickets]
command = "uv"
args = ["run", "--directory", "/absolute/path/to/cheq", "python", "-m", "server"]
```

Use an absolute path — the server resolves its data relative to its own source
file, so it starts correctly from any working directory.

## The tools

| tool | answers |
|---|---|
| `get_schema` | What columns exist, what values they take, and **what this data cannot answer** |
| `run_sql` | One read-only `SELECT`. Counts, mixes, distributions |
| `find_similar_tickets` | Nearest past tickets and their replies — or an explicit `no_precedent` |
| `suggest_routing` | Predicted queue and priority, with how far to trust the prediction |

## Try it

Ask your client these, in roughly this order:

1. *"What's in this ticket dataset, and what can't it tell me?"* — calls `get_schema`.
2. *"Which tags show up most in high-priority billing tickets?"* — `run_sql`. Note
   it counts with `COUNT(DISTINCT ticket_id)`; the schema warns that joining tags
   multiplies rows.
3. *"How have we handled this before?"* with a pasted ticket — `find_similar_tickets`
   returns past tickets **and the replies that were sent**.
4. *"Run find_similar_tickets on this text: How long should I braise a beef
   shank?"* — returns `has_precedent: false` instead of a confident draft. Note the
   explicit instruction: asked plainly, Claude answers from its own knowledge and
   never calls the tool, which is correct. **Abstention governs what the server
   returns, not whether the host consults it** — a tool cannot stop a model
   answering from what it already knows.
5. *"Which agent closes the most tickets?"* — refused. There is no assignee column,
   and `get_schema` says so up front. This one the model *does* reach for the tool,
   because the question is about the data.

## What it does with the data

```
3 source CSVs ──► 8 pipeline stages ──► DuckDB + 31,625-vector index
  61,765 rows        ~80s, deterministic     40,047 tickets, 10 queues
```

The stages are numbered and each prints a reconciliation, so you can run one at a
time and check the arithmetic:

```bash
uv run python pipeline/03_filter.py
```

Notable: duplicates are removed **before** the train/test split (~28% of a naive
test set would otherwise have its exact twin in training), and the DuckDB schema
deliberately omits the positional `tag_1..tag_8` columns so that a wrong
`GROUP BY` is unwritable rather than merely discouraged. The source's own language
label is wrong on 10% of rows — 4,204 tickets marked German whose text is English —
so `language` is re-detected from the text and the original kept as
`language_label`. Tag spellings are folded (`Data Breach` / `Data breach`,
`access control` / `access_control`) and the 62 cells that held a comma-joined
list are split, so `GROUP BY tag` counts one tag once.

## Evaluation

`make eval` runs offline in ~30s and writes [`reports/routing.md`](reports/routing.md)
and [`reports/labels.md`](reports/labels.md).

- **Routing: 0.707 macro-F1** across 10 queues, against a 0.045 majority-class baseline
  (English 0.782, German 0.542)
- **Abstention:** accepts 97.8% of real tickets, rejects 100% of 20 off-topic queries
  — see [`reports/calibration.md`](reports/calibration.md)

Every threshold the server uses is derived by these scripts and written to
`data/interim/`, not hardcoded. The abstention floor, the number of neighbours to
vote over, and the confidence bands are all measured on a validation split and
reported on a held-out test split.

## Layout

```
pipeline/   8 numbered stages + embedding.py (shared with the server)
server/     __main__.py (tools) · db.py (SQL guard) · retrieval.py (vector search)
evals/      run_routing.py · abstention_queries.yaml · answer_labels.yaml
reports/    generated: calibration.md · routing.md · labels.md
docs/       design.pdf — the one-page design note
```

## Known limits

- The corpus is **synthetic**, generated for classifier training. It demonstrates a
  mechanism; it is not evidence about real customers.
- **About half of all first replies are dead ends** — they neither resolve the ticket
  nor tell the customer what to send, so each one costs a round trip and buys nothing
  (49.8% over 39,739 labelled answers, 55% over 40 read by hand — `reports/labels.md`).
  `find_similar_tickets` returns `reply_state` on every hit so a draft can copy the
  replies that ended the round trip rather than the ones that caused it.
- **7.6% of answers are not support replies at all** — customer text sitting in the
  answer column.
- The four-way `answer_kind` label agrees with hand adjudication only **70%**; only
  the collapsed `reply_state` should be quoted. `reports/labels.md` says why.
- Hand-typed questions score lower than real tickets regardless of topic, so
  `mode="precedent"` wrongly refuses about 40% of legitimate typed queries.
  Use `mode="explore"` for those. The measurement is in `reports/calibration.md`.
- German routing trails English by 0.24 macro-F1.
