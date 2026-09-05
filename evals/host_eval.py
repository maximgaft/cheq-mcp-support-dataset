"""Host-driven evaluation: a real model answers the question set through the MCP tools.

The other evals test the server. This one tests the promise the server makes to a
user: a natural-language question goes in, a host LLM chooses tools, writes SQL,
reads what comes back, and answers. Each question runs as a fresh conversation
against the live MCP server, started the way .mcp.json starts it.

Needs an Anthropic API key (the only eval that does), so it is not part of
`make eval`. Run it with `make host-eval`; the report is committed so a reader
without a key still sees question -> tools -> answer for a named model.

Grading is deterministic and deliberately simple: a SQL question passes when
every expected value appears in the final answer and a tool was called (a more
precise number that rounds to the key counts); a refusal question passes when its
first sentence refuses and the answer names the missing data. Two open-ended
prompts are recorded but not graded. The full trajectories and the graded rows go
to data/interim/, so `--regrade` can re-score a run without another API call.

    uv run python evals/host_eval.py                # claude-opus-5, one pass
    uv run python evals/host_eval.py --model claude-sonnet-5 --reps 3
    uv run python evals/host_eval.py --dry-run      # no API call: list tools and questions
    uv run python evals/host_eval.py --regrade      # re-grade the last run from its saved answers
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
ROWS = ROOT / "data" / "interim" / "host_eval_rows.json"   # everything but the transcript, for --regrade

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
    if isinstance(expected, bool):
        return str(expected).lower() in answer.lower()
    if isinstance(expected, float) and not float(expected).is_integer():
        if number_pattern(expected).search(answer):
            return True
        # 39.72 answers a key of 39.7: accept any number that rounds to the key at its precision
        decimals = len(f"{expected:g}".split(".")[1])
        for token in re.findall(r"(?<![\d.])\d[\d,]*\.\d+", answer):
            try:
                if round(float(token.replace(",", "")), decimals) == expected:
                    return True
            except ValueError:
                pass
        return False
    if isinstance(expected, (int, float)):
        return bool(number_pattern(expected).search(answer))
    return str(expected).lower() in answer.lower()


def first_sentence(text: str) -> str:
    text = re.sub(r"[*_`#]", "", text).strip()
    return re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0]


def grade(case: dict, answer: str, tools_called: list[str]) -> tuple[bool, str]:
    """(passed, reason). Deterministic; a human can overrule from the recorded answer."""
    if case["kind"] == "sql":
        expected = case.get("expect_rows") or [case["expect"]]
        missing = [v for row in expected for v in row if not contains(answer, v)]
        if missing:
            return False, f"missing {missing}"
        if not tools_called:
            return False, "right numbers but no tool was called"
        return True, "every expected value present" + ("" if "run_sql" in tools_called else " (from get_schema, no SQL needed)")
    # refusal: the missing data is named and no confident number is given
    key = case["expect"].split()[-1]          # assignee / timestamps / identifier
    terms = MISSING_DATA_TERMS.get(key, (key.rstrip("s"),))
    named = any(term in answer.lower() for term in terms)
    # The refusal has to lead: a model that answers first and hedges later is not refusing.
    lead = first_sentence(answer).lower()
    refused = any(marker in lead for marker in REFUSAL_MARKERS)
    if named and refused:
        return True, f"refused up front and named the missing data ({key})"
    if refused:
        return False, f"refused but did not name the missing data ({key})"
    return False, "did not lead with a refusal"


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
    parser.add_argument("--regrade", action="store_true", help="re-grade the saved answers of the last run; no server, no API call")
    args = parser.parse_args()

    cases = yaml.safe_load(QUESTIONS.read_text())
    if args.regrade:
        regrade({c["id"]: c for c in cases})
        return
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
    ROWS.write_text(json.dumps({"model": args.model, "rows": [{k: v for k, v in r.items() if k != "transcript"} for r in rows]},
                               indent=1, default=str))
    print(f"  wrote {REPORT.relative_to(ROOT)}, {ROWS.relative_to(ROOT)} and {TRANSCRIPTS.relative_to(ROOT)}")


def regrade(cases: dict) -> None:
    """Re-apply grade() to the saved answers of the last run and rewrite the report.

    Uses the rows file when the run wrote one; otherwise rebuilds rows from the
    transcripts, in which case token usage is unknown and left out."""
    if ROWS.exists():
        saved = json.loads(ROWS.read_text())
        model, rows = saved["model"], saved["rows"]
    else:
        model, rows = "claude-opus-5", []
        for t in json.loads(TRANSCRIPTS.read_text()):
            case = cases.get(t["id"]) or next(({**u, "kind": "ungraded"} for u in UNGRADED if u["id"] == t["id"]), None)
            turns = [m for m in t["transcript"] if m["role"] == "assistant"]
            trace = [{"tool": b["name"], "input": b["input"], "is_error": False, "result_chars": 0}
                     for m in turns for b in m["content"] if b.get("type") == "tool_use"]
            answer = " ".join(b["text"] for b in turns[-1]["content"] if b.get("type") == "text") if turns else ""
            rows.append({"rep": t["rep"], "id": t["id"], "kind": case["kind"], "question": case["question"],
                         "answer": answer, "trace": trace, "usage": None, "served_model": None,
                         "stop_reason": "end_turn", "seconds": None})
    for r in rows:
        r["tools"] = [t["tool"] for t in r["trace"]]
        case = cases.get(r["id"])
        if r["kind"] == "ungraded" or case is None:
            r["passed"], r["reason"] = None, "recorded, not graded"
        elif r["stop_reason"] not in ("end_turn", "tool_use"):
            r["passed"], r["reason"] = None, f"not scored: stop_reason={r['stop_reason']}"
        else:
            r["passed"], r["reason"] = grade(case, r["answer"], r["tools"])
        print(f"  {({True: 'pass', False: 'FAIL', None: '----'}[r['passed']])}  {r['id']:<28} {r['reason']}")
    write_report(model, rows, regraded=True)
    print(f"  wrote {REPORT.relative_to(ROOT)} (re-graded from the saved run)")


def write_report(model: str, rows: list[dict], regraded: bool = False) -> None:
    import anthropic
    graded = [r for r in rows if r["kind"] != "ungraded" and r["passed"] is not None]
    passed = sum(1 for r in graded if r["passed"])
    served = sorted({r["served_model"] for r in rows if r["served_model"]}) or [model]
    known = [r["usage"] for r in rows if r["usage"]]
    usage = {k: sum(u[k] for u in known) for k in ("input", "output", "cache_read", "cache_write")} if known else None
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
        "passes when every expected value appears in the answer and a tool was called; a refusal "
        "question passes when its first sentence refuses and the answer names the missing data. Two "
        "open-ended prompts are recorded below without a grade."
        + (f" Tokens: {usage['input']:,} uncached in, {usage['cache_read']:,} read from cache, "
           f"{usage['cache_write']:,} written to cache, {usage['output']:,} out." if usage else "")
        + (" Grades were recomputed from the saved run after a grader fix; the answers are unchanged." if regraded else ""),
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
        timing = f"{r['seconds']}s, " if r.get("seconds") is not None else ""
        lines.append(f"**{r['id']}** ({timing}stop `{r['stop_reason']}`)")
        for t in r["trace"]:
            arg = t["input"].get("query") or t["input"].get("text") or json.dumps(t["input"])
            arg = " ".join(str(arg).split())
            arg = arg if len(arg) <= 200 else arg[:200] + " ..."
            size = f" → {t['result_chars']:,} chars" if t.get("result_chars") else ""
            lines.append(f"- `{t['tool']}` {'(error) ' if t['is_error'] else ''}{arg}{size}")
        lines.append("")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines))


if __name__ == "__main__":
    asyncio.run(main())
