"""Out-of-band pass: label what each first reply actually did.

Unnumbered on purpose. Every numbered stage runs offline and deterministically;
this one needs ANTHROPIC_API_KEY and costs money, so it is run once by hand and
its output is committed. `make build` then reads the committed file and stays
offline for anyone cloning the repo.

Two fields per ticket:

  kind             resolution / information_request / acknowledgement /
                   not_a_support_reply
  ask_is_specific  for information requests only - does the reply name something
                   the customer can actually go and fetch, or just ask for
                   "details"?

The second field is the one worth paying for. A first reply that says "please
provide details" guarantees another round trip, because the customer cannot tell
what to send, and elapsed time is the expensive thing in a support queue. It is
also a judgement that needs reading: "please furnish the system logs" and "please
share details about your current campaigns" are the same shape and opposite in
value.

Why an LLM rather than a regex: I estimated this split by hand twice and got 33%
then 75%, the difference being how much of each answer I had truncated. A rule
that keys on request phrasing cannot separate a named artefact from a named topic.

Safety gate: the run always labels the 40 adjudicated tickets in
evals/answer_labels.yaml first and prints agreement. Nothing is spent on the full
corpus unless you pass --all, so a broken prompt costs about a cent.

  uv run python pipeline/label.py            # gold set only, ~1 cent
  uv run python pipeline/label.py --all      # full corpus, ~$10
"""

from __future__ import annotations

import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "data" / "interim" / "05_split.parquet"
GOLD = ROOT / "evals" / "answer_labels.yaml"
OUT = ROOT / "data" / "answer_labels.parquet"

MODEL = "claude-haiku-4-5"
BATCH = 25          # answers per request - amortises the instructions ~25x
CONCURRENCY = 8
MAX_ANSWER_CHARS = 1200
KINDS = {"resolution", "information_request", "acknowledgement", "not_a_support_reply"}

SYSTEM = """You label the FIRST REPLY a support agent sent to a customer's ticket.

Return two fields per item.

`kind` - exactly one of:
  resolution           answers substantively; the customer can act without
                       sending anything back
  information_request  cannot proceed until the customer supplies something
  acknowledgement      received / looking into it / we will call - no substance
                       and no specific ask
  not_a_support_reply  not an agent's reply at all. Usually it reads customer-side
                       ("I am writing to report..."), or it tells the reporter to
                       go investigate their own ticket.

`ask_is_specific` - only when kind is information_request, otherwise null:
  true   the customer knows exactly what to fetch. "the system logs", "your
         account number and the invoice number", "the model of your KVM switch",
         "the exact date and time of discovery", "which platform you are using"
  false  names a topic, not a thing. "details about your current campaigns and
         objectives", "mehr Details oder relevante Informationen", "the details
         of the issue you are facing"

A reply that asks for several things counts as specific if any one of them is a
named artefact. Replies are English or German; label both the same way.

Return ONLY a JSON array, one object per input, same order, no prose:
[{"n": 1, "kind": "...", "ask_is_specific": true}, ...]"""


def collapse(kind: str, specific: bool | None) -> str:
    """Can the customer act on this reply alone? All the business claim needs.

    The four-way `kind` agrees with the adjudicated set only 70% of the time -
    `acknowledgement` is where it scatters. This collapse agrees 75% and, more
    importantly, is distributionally unbiased: the dominant confusion moves
    tickets *within* dead_end, since a vague ask and a bare acknowledgement are
    both dead ends. See reports/labels.md.
    """
    if kind == "resolution":
        return "resolved"          # no round trip required
    if kind == "information_request" and specific:
        return "actionable_ask"    # a round trip, but a productive one
    return "dead_end"              # vague ask, bare acknowledgement, or not a reply


def parse(text: str, expected: int) -> list[dict]:
    """Tolerant extraction - pull the first JSON array out of the response."""
    match = re.search(r"\[.*\]", text, re.S)
    if not match:
        raise ValueError(f"no JSON array in response: {text[:120]!r}")
    items = json.loads(match.group(0))
    if len(items) != expected:
        raise ValueError(f"expected {expected} labels, got {len(items)}")
    for item in items:
        if item.get("kind") not in KINDS:
            raise ValueError(f"unknown kind {item.get('kind')!r}")
        if item["kind"] != "information_request":
            item["ask_is_specific"] = None
    return items


def label_batch(client, rows: pd.DataFrame) -> list[dict]:
    numbered = "\n\n".join(
        f"[{n}] {' '.join(str(a).split())[:MAX_ANSWER_CHARS]}"
        for n, a in enumerate(rows.answer, 1)
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=SYSTEM,
        messages=[{"role": "user", "content": numbered}],
    )
    text = "".join(b.text for b in response.content if b.type == "text")
    items = parse(text, len(rows))
    usage = (response.usage.input_tokens, response.usage.output_tokens)
    return [
        {"ticket_id": t, "kind": i["kind"], "ask_is_specific": i["ask_is_specific"],
         "_usage": usage if n == 0 else None}
        for n, (t, i) in enumerate(zip(rows.ticket_id, items))
    ]


def run(client, frame: pd.DataFrame) -> pd.DataFrame:
    chunks = [frame.iloc[i : i + BATCH] for i in range(0, len(frame), BATCH)]
    print(f"  {len(frame):,} answers in {len(chunks)} requests, {CONCURRENCY} at a time")
    out, failures = [], 0
    with ThreadPoolExecutor(CONCURRENCY) as pool:
        for done, result in enumerate(pool.map(lambda c: _safe(client, c), chunks), 1):
            if result is None:
                failures += 1
            else:
                out += result
            if done % 10 == 0 or done == len(chunks):
                print(f"    {done}/{len(chunks)} requests, {failures} failed", flush=True)
    if failures:
        print(f"  WARNING: {failures} requests failed and those tickets are unlabelled")

    tokens_in = sum(r["_usage"][0] for r in out if r["_usage"])
    tokens_out = sum(r["_usage"][1] for r in out if r["_usage"])
    print(f"  tokens: {tokens_in:,} in / {tokens_out:,} out "
          f"-> ${tokens_in / 1e6 * 1.0 + tokens_out / 1e6 * 5.0:.2f} at Haiku 4.5 rates")
    return pd.DataFrame(out).drop(columns="_usage")


def _safe(client, chunk):
    try:
        return label_batch(client, chunk)
    except Exception as exc:  # noqa: BLE001 - one bad batch must not lose the run
        print(f"    batch failed: {type(exc).__name__}: {exc}", flush=True)
        return None


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. This is the only step that needs it.")
    import anthropic

    client = anthropic.Anthropic()
    corpus = pd.read_parquet(IN)
    indexable = corpus[corpus.is_indexable]

    # --- gate: the adjudicated set first, always ---
    gold = {c["id"]: c for c in yaml.safe_load(GOLD.read_text())}
    gold_rows = indexable[indexable.ticket_id.isin(gold)].reset_index(drop=True)
    print(f"gold set ({len(gold_rows)} adjudicated tickets)")
    predicted = run(client, gold_rows).set_index("ticket_id")

    kind_hits = sum(predicted.loc[i, "kind"] == g["label"] for i, g in gold.items()
                    if i in predicted.index)
    spec = [(i, g) for i, g in gold.items()
            if "ask_is_specific" in g and i in predicted.index
            and predicted.loc[i, "kind"] == "information_request"]
    spec_hits = sum(predicted.loc[i, "ask_is_specific"] == g["ask_is_specific"] for i, g in spec)

    print(f"\n  kind agreement            {kind_hits}/{len(gold)} "
          f"({100 * kind_hits / len(gold):.0f}%)")
    print(f"  ask_is_specific agreement {spec_hits}/{len(spec)}"
          + (f" ({100 * spec_hits / len(spec):.0f}%)" if spec else " (no overlap)"))
    print("\n  disagreements:")
    for i, g in gold.items():
        if i in predicted.index and predicted.loc[i, "kind"] != g["label"]:
            print(f"    {i:<22} gold={g['label']:<20} model={predicted.loc[i, 'kind']}")

    if "--all" not in sys.argv:
        print("\nGold set only. Re-run with --all to label the full corpus.")
        return

    print(f"\nfull corpus ({len(indexable):,} indexable tickets)")
    labels = run(client, indexable.reset_index(drop=True))
    labels.to_parquet(OUT, index=False)
    print(f"\n  {labels.kind.value_counts().to_dict()}")
    ir = labels[labels.kind.eq("information_request")]
    if len(ir):
        print(f"  of {len(ir):,} information requests, "
              f"{100 * ir.ask_is_specific.eq(False).mean():.1f}% ask for nothing specific")
    print(f"  wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
