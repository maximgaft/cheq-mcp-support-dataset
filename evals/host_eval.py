"""Host-driven evaluation: a real model answers the question set through the MCP tools.

The other evals test the server. This one tests the promise the server makes to a
user: a natural-language question goes in, a host LLM chooses tools, writes SQL,
reads what comes back, and answers. Each question runs as a fresh conversation
against the live MCP server, started the way .mcp.json starts it.

Needs an Anthropic API key (the only eval that does), so it is not part of
`make eval`. Run it with `make host-eval`; the report is committed so a reader
without a key still sees question -> tools -> answer for a named model.

Grading is deterministic and deliberately simple: a SQL question passes when
every expected value appears in the final answer; a refusal question passes when
the answer names the missing data and gives no number. Two open-ended prompts are
recorded but not graded. The full trajectories go to data/interim/ for debugging.

    uv run python evals/host_eval.py                # claude-opus-5, one pass
    uv run python evals/host_eval.py --model claude-sonnet-5 --reps 3
    uv run python evals/host_eval.py --dry-run      # no API call: list tools and questions
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import date
from pathlib import Path

import yaml
from mcp import ClientSession
from mcp.client.stdio import (
    StdioServerParameters,
    get_default_environment,
    stdio_client,
)

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "evals" / "questions.yaml"
REPORT = ROOT / "reports" / "host_eval.md"
TRANSCRIPTS = ROOT / "data" / "interim" / "host_eval_transcripts.json"

DEFAULT_MODEL = "claude-opus-5"
MAX_TURNS = 8            # tool round-trips per question before we call it a loop
MAX_TOKENS = 4096        # answers are a sentence or a small table
CALL_TIMEOUT = 300.0     # seconds per MCP tool call (first retrieval call loads the model)

# Open-ended prompts from the assessment packet: recorded, not graded.
UNGRADED = [
    {"id": "route_german_ticket",
     "question": "Route this new short German ticket: \"Meine Rechnung wurde doppelt abgebucht. "
                 "An wen soll ich mich wenden?\" Then explain the evidence and the uncertainty."},
    {"id": "resolved_label_meaning",
     "question": "A retrieved reply is labelled resolved. Can we tell the customer that this "
                 "solution worked?"},
]

# What counts as naming the missing data, per refusal case.
MISSING_DATA_TERMS = {
    "assignee": ("assignee", "agent"),
    "timestamps": ("timestamp", "date", "time"),
    "identifier": ("identifier", "customer id", "customer-level", "per customer", "per-customer",
                   "individual customer", "which customer"),
}

REFUSAL_MARKERS = ("cannot", "can't", "can not", "isn't", "aren't", "doesn't", "don't", "won't",
                   "not ", "no ", "unable", "unavailable", "missing", "lack")


# ----------------------------------------------------------------------------- grading
def number_pattern(value: float | int) -> re.Pattern:
    """Match a number as a model would write it: 40047, 40,047, 34.0, 34.0%, 34%."""
    if isinstance(value, float) and not float(value).is_integer():
        text = f"{value:g}"
        return re.compile(r"(?<![\d.])" + re.escape(text) + r"(?!\d)")
    n = int(value)
    return re.compile(r"(?<![\d.])" + f"{n:,}".replace(",", ",?") + r"(?![\d,]\d)")


def contains(answer: str, expected) -> bool:
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return bool(number_pattern(expected).search(answer))
    return str(expected).lower() in answer.lower()


def grade(case: dict, answer: str, tools_called: list[str]) -> tuple[bool, str]:
    """(passed, reason). Deterministic; a human can overrule from the recorded answer."""
    if case["kind"] == "sql":
        expected = case.get("expect_rows") or [case["expect"]]
        missing = [v for row in expected for v in row if not contains(answer, v)]
        if missing:
            return False, f"missing {missing}"
        if "run_sql" not in tools_called:
            return False, "right numbers but run_sql was never called"
        return True, "every expected value present"
    # refusal: the missing data is named and no confident number is given
    key = case["expect"].split()[-1]          # assignee / timestamps / identifier
    terms = MISSING_DATA_TERMS.get(key, (key.rstrip("s"),))
    named = any(term in answer.lower() for term in terms)
    refused = any(marker in answer.lower() for marker in REFUSAL_MARKERS)
    if re.search(r"\d{2,}", answer):
        return False, "gave a number instead of refusing"
    if named and refused:
        return True, f"refused and named the missing data ({key})"
    if refused:
        return False, f"refused but did not name the missing data ({key})"
    return False, "did not refuse"


# ----------------------------------------------------------------------------- the loop
def mcp_tools_to_anthropic(tools) -> list[dict]:
    return [{"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
            for t in tools]


async def ask(client, model: str, system: str, tools: list[dict], session, question: str) -> dict:
    """One fresh conversation. Returns the answer, the tool trace, usage and stop reason."""
    messages = [{"role": "user", "content": question}]
    trace, usage = [], {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
    served_model, stop_reason, started = None, None, time.time()
    for _ in range(MAX_TURNS):
        response = await client.messages.create(
            model=model, max_tokens=MAX_TOKENS, system=system, tools=tools, messages=messages,
            cache_control={"type": "ephemeral"},
        )
        served_model = response.model
        usage["input"] += response.usage.input_tokens
        usage["output"] += response.usage.output_tokens
        usage["cache_read"] += response.usage.cache_read_input_tokens or 0
        usage["cache_write"] += response.usage.cache_creation_input_tokens or 0
        stop_reason = response.stop_reason
        messages.append({"role": "assistant", "content": response.content})
        if stop_reason != "tool_use":
            break
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result = await session.call_tool(block.name, block.input, read_timeout_seconds=CALL_TIMEOUT)
            text = result.content[0].text if result.content else ""
            trace.append({"tool": block.name, "input": block.input, "is_error": bool(result.is_error),
                          "result_chars": len(text)})
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": text,
                            "is_error": bool(result.is_error)})
        messages.append({"role": "user", "content": results})   # all results in one message
    answer = " ".join(b.text for b in messages[-1]["content"] if getattr(b, "type", None) == "text") \
        if messages[-1]["role"] == "assistant" else ""
    return {"answer": answer, "trace": trace, "usage": usage, "served_model": served_model,
            "stop_reason": stop_reason, "seconds": round(time.time() - started, 1),
            "transcript": [m if isinstance(m["content"], str) else
                           {"role": m["role"], "content": [c if isinstance(c, dict) else c.to_dict()
                                                            for c in m["content"]]}
                           for m in messages]}


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=os.environ.get("CHEQ_HOST_MODEL", DEFAULT_MODEL))
    parser.add_argument("--reps", type=int, default=1, help="passes over the question set")
    parser.add_argument("--dry-run", action="store_true", help="start the server, list tools and questions, make no API call")
    args = parser.parse_args()

    cases = yaml.safe_load(QUESTIONS.read_text())
    if not args.dry_run and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. This is the only eval that needs it; "
                 "use --dry-run to check the wiring without a call.")

    env = {**get_default_environment(), **{k: os.environ[k] for k in ("CHEQ_DB", "HF_HOME", "HF_HUB_OFFLINE") if k in os.environ}}
    params = StdioServerParameters(command="uv", args=["run", "--directory", str(ROOT), "python", "-m", "server"], env=env)
    async with stdio_client(params, errlog=open(os.devnull, "w")) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            tools = mcp_tools_to_anthropic((await session.list_tools()).tools)
            system = ((init.instructions or "") + "\n\nAnswer the user's question with the tools. "
                      "Give the final answer plainly, with the numbers. If the data cannot answer "
                      "the question, say so and say what is missing rather than estimating.")
            if args.dry_run:
                print(f"server: {init.server_info.name} | tools: {[t['name'] for t in tools]}")
                print(f"model: {args.model} | graded: {len(cases)} | ungraded: {len(UNGRADED)}")
                for c in cases:
                    print(f"  [{c['kind']:6}] {c['question']}")
                print("dry run: no API call made")
                return

            import anthropic
            client = anthropic.AsyncAnthropic()
            rows, transcripts = [], []
            for rep in range(args.reps):
                for case in cases + [{**u, "kind": "ungraded"} for u in UNGRADED]:
                    out = await ask(client, args.model, system, tools, session, case["question"])
                    tools_called = [t["tool"] for t in out["trace"]]
                    if case["kind"] == "ungraded":
                        passed, reason = None, "recorded, not graded"
                    elif out["stop_reason"] not in ("end_turn", "tool_use"):
                        passed, reason = None, f"not scored: stop_reason={out['stop_reason']}"
                    else:
                        passed, reason = grade(case, out["answer"], tools_called)
                    mark = {True: "pass", False: "FAIL", None: "----"}[passed]
                    print(f"  {mark}  {case['id']:<28} {reason}  [{', '.join(tools_called) or 'no tools'}]")
                    rows.append({"rep": rep, "id": case["id"], "kind": case["kind"], "question": case["question"],
                                 "passed": passed, "reason": reason, "tools": tools_called, **out})
                    transcripts.append({"rep": rep, "id": case["id"], "transcript": out["transcript"]})

    write_report(args.model, rows)
    TRANSCRIPTS.parent.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTS.write_text(json.dumps(transcripts, indent=1, default=str))
    print(f"  wrote {REPORT.relative_to(ROOT)} and {TRANSCRIPTS.relative_to(ROOT)}")


def write_report(model: str, rows: list[dict]) -> None:
    import anthropic
    graded = [r for r in rows if r["kind"] != "ungraded" and r["passed"] is not None]
    passed = sum(1 for r in graded if r["passed"])
    served = sorted({r["served_model"] for r in rows if r["served_model"]})
    usage = {k: sum(r["usage"][k] for r in rows) for k in ("input", "output", "cache_read", "cache_write")}
    reps = max((r["rep"] for r in rows), default=0) + 1
    lines = [
        "# Host-driven evaluation: a model answering through the MCP tools",
        "",
        f"Generated by `evals/host_eval.py` on {date.today().isoformat()}. Model requested `{model}`, "
        f"served `{', '.join(served)}`, `anthropic` SDK {anthropic.__version__}. Each question ran as a "
        f"fresh conversation against the server started as `.mcp.json` starts it, with the server's own "
        f"instructions as the system prompt plus one sentence asking for a plain answer. {reps} pass(es).",
        "",
        f"**{passed} of {len(graded)} graded answers pass.** Grading is deterministic: a SQL question "
        "passes when every expected value appears in the answer and `run_sql` was called; a refusal "
        "question passes when the answer names the missing data and refuses. Two open-ended prompts are "
        "recorded below without a grade. Tokens: "
        f"{usage['input']:,} in ({usage['cache_read']:,} from cache), {usage['output']:,} out.",
        "",
        "| question | tools called | result | answer (trimmed) |",
        "|---|---|---|---|",
    ]
    for r in rows:
        mark = {True: "pass", False: "**FAIL**", None: "—"}[r["passed"]]
        answer = " ".join(r["answer"].split())
        answer = answer if len(answer) <= 220 else answer[:220].rstrip() + " ..."
        tools = ", ".join(f"`{t}`" for t in r["tools"]) or "none"
        lines.append(f"| {r['question']} | {tools} | {mark} ({r['reason']}) | {answer.replace('|', '/')} |")
    lines += ["", "## Tool traces", ""]
    for r in rows:
        lines.append(f"**{r['id']}** ({r['seconds']}s, stop `{r['stop_reason']}`)")
        for t in r["trace"]:
            arg = t["input"].get("query") or t["input"].get("text") or json.dumps(t["input"])
            arg = " ".join(str(arg).split())
            arg = arg if len(arg) <= 200 else arg[:200] + " ..."
            lines.append(f"- `{t['tool']}` {'(error) ' if t['is_error'] else ''}{arg} → {t['result_chars']:,} chars")
        lines.append("")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines))


if __name__ == "__main__":
    asyncio.run(main())
