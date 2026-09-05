"""End-to-end smoke test over MCP stdio.

Starts the server exactly as .mcp.json does, lists the tools, calls each one the
way a host would, checks the shape of every answer, and writes the transcript to
reports/smoke.md - so a reader who never installs anything can see question ->
tool -> answer. Exits 1 on any unexpected shape.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import tempfile
from pathlib import Path

from huggingface_hub import try_to_load_from_cache
from mcp import ClientSession
from mcp.client.stdio import (
    StdioServerParameters,
    get_default_environment,
    stdio_client,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.embedding import MODEL_NAME  # noqa: E402

REPORT = ROOT / "reports" / "smoke.md"
CALL_TIMEOUT = 600.0   # seconds; the first retrieval call may load, or download, the model
FORWARDED_ENV = ("CHEQ_DB", "HF_HOME", "HF_HUB_OFFLINE")
EXPECTED_TOOLS = {
    "get_schema": [],
    "run_sql": ["query", "max_rows"],
    "find_similar_tickets": ["text", "queue", "language", "k"],
    "suggest_routing": ["text"],
}
BILLING = ("Billing discrepancy on latest invoice. I was charged twice for my monthly "
           "subscription in March. My account number is <acc_num>. Please refund the "
           "duplicate charge and confirm my billing cycle.")
OFF_TOPIC = "How long should I braise a beef shank, and do I need to sear it first?"
FRAGMENT = "my wifi keeps dropping"


KEEP_WHOLE = {"guidance", "error", "cannot_answer", "why"}   # the server's own words: never shortened


def trim(value, limit: int = 160):
    """Shorten long ticket text so the transcript stays readable."""
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit].rstrip() + " ..."
    if isinstance(value, list):
        return [trim(v, limit) for v in value]
    if isinstance(value, dict):
        return {k: (v if k in KEEP_WHOLE else trim(v, limit)) for k, v in value.items()}
    return value


class SmokeFailure(Exception):
    """A check did not hold. Raised inside the session, reported outside it."""


def check(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeFailure(message)
    print(f"  ok   {message}")


def root_cause(log_text: str) -> list[str]:
    """The exception lines from the server's rich-formatted traceback, without the boxes."""
    lines = [re.sub(r"[│╭╮╰╯─❱]", "", line).strip() for line in log_text.splitlines()]
    lines = [line for line in lines if line and "HTTP Request" not in line]
    starts = [i for i, line in enumerate(lines) if re.match(r"^[A-Z]\w*(Error|Exception)\b", line)]
    if not starts:
        return lines[-8:]
    first, last = starts[0], starts[-1]
    out = lines[first:first + 4]
    if last != first:
        out += ["...", lines[last]]
    return out


def find(exc: BaseException, kind: type) -> BaseException | None:
    """The first exception of `kind` inside a (possibly nested) exception group -
    anyio wraps anything raised inside the stdio session in one."""
    if isinstance(exc, kind):
        return exc
    if isinstance(exc, BaseExceptionGroup):
        for sub in exc.exceptions:
            if (hit := find(sub, kind)) is not None:
                return hit
    return None


async def main() -> None:
    env = {**get_default_environment(), **{k: os.environ[k] for k in FORWARDED_ENV if k in os.environ}}
    params = StdioServerParameters(
        command="uv", args=["run", "--directory", str(ROOT), "python", "-m", "server"], env=env)
    if not isinstance(try_to_load_from_cache(MODEL_NAME, "config.json"), str):
        print(f"  note: {MODEL_NAME} is not cached yet - the first retrieval call downloads it once (~470 MB)")
    transcript = []

    # The server's stderr goes to a log that is shown only if something fails.
    log_path = Path(tempfile.gettempdir()) / "cheq-smoke-server.log"
    with log_path.open("w") as errlog:
        try:
            await run_calls(params, errlog, transcript)
        except BaseException as exc:
            errlog.flush()
            cause = root_cause(log_path.read_text())
            if cause:
                print("  server said:\n    " + "\n    ".join(cause), file=sys.stderr)
            failure = find(exc, SmokeFailure)
            if failure is None:
                raise
            sys.exit(f"  SMOKE FAILED: {failure}")
    write_report(transcript)


async def run_calls(params, errlog, transcript) -> None:
    async with stdio_client(params, errlog=errlog) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            check(init.server_info.name == "cheq-tickets", f"server is {init.server_info.name}")

            tools = {t.name: list(t.input_schema.get("properties", {})) for t in (await session.list_tools()).tools}
            check(tools == EXPECTED_TOOLS, f"four tools with the expected parameters: {sorted(tools)}")

            async def call(name: str, **args):
                result = await session.call_tool(name, args, read_timeout_seconds=CALL_TIMEOUT)
                text = result.content[0].text if result.content else ""
                if result.is_error:
                    raise SmokeFailure(f"{name} returned an error: {text}")
                payload = result.structured_content or json.loads(text)
                if set(payload) == {"result"}:
                    payload = payload["result"]
                transcript.append((name, args, payload))
                return payload

            schema = await call("get_schema")
            served = schema["source_rows"]["tickets_served"]
            check(served == schema["tables"]["tickets"]["rows"],
                  f"get_schema: {served:,} tickets served, provenance reconciles")

            top = await call("run_sql", query="SELECT queue, count(*) AS n FROM tickets GROUP BY 1 ORDER BY 2 DESC LIMIT 3")
            first = schema["categorical_values"]["queue"][0]
            check(top["rows"][0] == [first["value"], first["tickets"]],
                  f"run_sql: largest queue {top['rows'][0][0]} = {top['rows'][0][1]:,}")

            chained = await call("run_sql", query="SELECT 1; SELECT 2")
            check("error" in chained and "one statement" in chained["error"], "run_sql: chained statements rejected")

            miss = await call("find_similar_tickets", text=OFF_TOPIC, k=2)
            check(miss["has_precedent"] is False and "guidance" in miss,
                  f"find_similar_tickets: off-topic text has no precedent (top {miss['top_similarity']})")

            hit = await call("find_similar_tickets", text=BILLING, k=3)
            check(hit["has_precedent"] is True and all("reply_state" in h for h in hit["results"]),
                  f"find_similar_tickets: billing ticket has precedent (top {hit['top_similarity']}), reply_state on every hit")

            low = await call("suggest_routing", text=FRAGMENT)
            check(low["expected_accuracy"] is None and "floor" in low["guidance"],
                  "suggest_routing: a fragment falls below the floor, confidence withheld")

            routed = await call("suggest_routing", text=BILLING)
            check(routed["queue"] == "Billing and Payments",
                  f"suggest_routing: billing ticket -> {routed['queue']} / {routed['priority']}")


def write_report(transcript) -> None:
    lines = ["# Smoke test: every tool over MCP stdio", "",
             "Generated by `evals/smoke.py`. The server was started the way `.mcp.json` starts it, "
             "and each call below went through the MCP protocol as a host would send it. To keep "
             "this readable, `get_schema` is shown as row counts, `source_rows` and `cannot_answer` "
             "only; each `find_similar_tickets` hit shows five of its ten fields; ticket text over "
             "160 characters is cut. Guidance and error strings are the server's exact words.", ""]
    for name, args, payload in transcript:
        if name == "get_schema":
            payload = {k: payload[k] for k in ("tables", "source_rows", "cannot_answer")}
            payload["tables"] = {t: v["rows"] for t, v in payload["tables"].items()}
        if name == "find_similar_tickets" and "results" in payload:
            payload["results"] = [{k: h[k] for k in ("ticket_id", "similarity", "queue", "reply_state", "answer")}
                                  for h in payload["results"]]
        lines += [f"## `{name}` {json.dumps(trim(args, 200))}", "", "```json",
                  json.dumps(trim(payload), indent=1, ensure_ascii=False), "```", ""]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines))
    print(f"  wrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    asyncio.run(main())
