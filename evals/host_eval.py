"""Host-driven evaluation: a real model answers the question set through the MCP tools.

The other evals test the server. This one tests the promise the server makes to a
user: a natural-language question goes in, a host LLM chooses tools, writes SQL,
reads what comes back, and answers. Each question runs as a fresh conversation
against the live MCP server, started the way .mcp.json starts it.

Needs an Anthropic API key (the only eval that does), so it is not part of
`make eval`. Run it with `make host-eval`; the report is committed so a reader
without a key still sees question -> tools -> answer for a named model.

Grading is deterministic. A SQL question passes when every expected value appears
in the answer, next to its own label when the answer is a table, and some tool was
called (a more precise number that rounds to the key counts). A refusal question
passes when its first sentence refuses, the missing data is named, and no trend is
asserted afterwards. Every number in an answer that no tool returned and that is
not a one-step sum, difference or ratio of returned numbers is listed as unsourced:
a warning for a human reader, not a fail, because the model may compute a share
correctly. Two open-ended prompts are recorded but not graded. Full trajectories
and graded rows go to data/interim/, so `--regrade` re-scores a run without a call.

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

# A refusal has to be a refusal phrase, not any stray "no" or "not": "Medium is the most
# common priority, not high" is an answer. The first sentence must contain one of these
# and no number that could be an answer.
REFUSAL_MARKERS = ("cannot", "can't", "can not", "unable", "not possible", "isn't possible",
                   "no way", "not available", "isn't available", "unavailable", "not recorded",
                   "not in the data", "not in this data", "does not contain", "doesn't contain",
                   "does not have", "doesn't have", "has no", "have no", "there is no", "there are no",
                   "is absent", "are absent", "missing", "lack", "not something", "can only")
NEGATION = ("cannot", "can't", "whether", "no way", "not ", "n't", "unable", "impossible")


# ----------------------------------------------------------------------------- grading
NUMBER = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s*(%|percent)?")
TREND = re.compile(r"\b(increased|decreased|grew|fell|rose|dropped|jumped|declined|doubled|halved)\b[^.]{0,40}\d"
                   r"|\d[\d,.]*\s*(?:%|percent)\s*(?:higher|lower|more|less|increase|decrease)", re.I)
SMALL_INT = 20        # integers below this are item counts (columns, queues, steps), not claims
WINDOW = 200          # characters around a label within which its row's values must sit


def numbers_in(text: str) -> list[tuple[str, float, bool]]:
    """(as written, value, is_percent) for every number in the text."""
    out = []
    for m in NUMBER.finditer(text):
        try:
            out.append((m.group(0).strip(), float(m.group(1).replace(",", "")), bool(m.group(2))))
        except ValueError:
            pass
    return out


def decimals_of(raw: str) -> int:
    body = raw.split("%")[0].replace("percent", "").strip().replace(",", "")
    return len(body.split(".")[1]) if "." in body else 0


def unsourced_numbers(answer: str, tool_texts: list[str], expected: list) -> list[str]:
    """Numbers in the answer that no tool returned and that are not derived from
    numbers the answer itself cites.

    A number is sourced if a tool returned it (or something that rounds to it) or it
    is an expected value. It is derived if it is a sum, difference or ratio of
    sourced numbers that appear in the answer, or of their one-step sums and
    differences - the model showing its working. Operands are restricted to the
    answer's own numbers on purpose: with a hundred numbers in a schema dump, some
    ratio rounds to almost any whole percentage, and the check would mean nothing."""
    sourced = {v for text in tool_texts for _, v, _ in numbers_in(text)}
    sourced |= {float(e) for e in expected if isinstance(e, (int, float)) and not isinstance(e, bool)}
    cited = [(raw, v, pct) for raw, v, pct in numbers_in(answer)]

    def is_sourced(value: float, k: int) -> bool:
        return any(round(s, k) == value for s in sourced)

    base = sorted({v for raw, v, _ in cited if is_sourced(v, decimals_of(raw)) and v >= 1})
    pairs = {a + b for i, a in enumerate(base) for b in base[i:]}
    triples = {a + b + c for i, a in enumerate(base) for j, b in enumerate(base[i:], i) for c in base[j:]}
    diffs = {abs(a - b) for a in base for b in base}
    totals = set(base) | pairs | triples | diffs        # what a model may have summed or subtracted
    parts = set(base) | pairs | diffs                   # what it may divide by a total
    flagged = []
    for raw, value, is_pct in cited:
        k = decimals_of(raw)
        if (not is_pct and value < SMALL_INT and value.is_integer()) or is_sourced(value, k):
            continue
        if any(round(t, k) == value for t in totals):
            continue
        if any(b and round(100 * a / b, k) == value for a in parts for b in totals):
            continue
        flagged.append(raw)
    return list(dict.fromkeys(flagged))


def label_spans(answer: str, label: str, others: list[str] = ()) -> list[int]:
    """Positions where `label` occurs as a whole word - and is not the start of a
    longer expected label ("IT" inside "IT Support" is not the IT row)."""
    flags = 0 if len(label) <= 3 else re.I      # "IT" must not match "it"
    longer = [o for o in others if len(o) > len(label)]
    return [m.start() for m in re.finditer(r"(?<!\w)" + re.escape(label) + r"(?!\w)", answer, flags)
            if not any(answer[m.start(): m.start() + len(o)].lower() == o.lower() for o in longer)]


def row_present(answer: str, row: list, other_labels: list[str]) -> bool:
    """Every value of the row appears, and next to the row's own label when it has one:
    the segment for a label runs from the previous other label to the next one."""
    labels = [v for v in row if isinstance(v, str)]
    values = [v for v in row if not isinstance(v, str)]
    if not labels:
        return all(contains(answer, v) for v in values)
    boundaries = sorted(pos for other in other_labels
                        for pos in label_spans(answer, other, [labels[0]] + [o for o in other_labels if o != other]))
    for start in label_spans(answer, labels[0], other_labels):
        lo = max([b for b in boundaries if b < start], default=start - WINDOW)
        hi = min([b for b in boundaries if b > start], default=start + WINDOW)
        if all(contains(answer[max(0, lo): hi], v) for v in values):
            return True
    return False


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


def sentences(text: str) -> list[str]:
    text = re.sub(r"[*_`#]", "", text).strip()
    return [part for part in re.split(r"(?<=[.!?])\s", text) if part]


def first_sentence(text: str) -> str:
    parts = sentences(text)
    return parts[0] if parts else ""


def leads_with_refusal(answer: str) -> bool:
    """The first sentence refuses: a bare "No", or a refusal phrase, and no number that
    could be an answer (a whole number of 20 or more, or any percentage)."""
    lead = first_sentence(answer).lower().strip()
    if re.match(r"^no\b[.,;:!]?$", lead) or re.match(r"^no[.,;:!]\s", lead):
        return True
    # "No timestamps are available", "Nothing here records ...", "There is no assignee ..."
    opens_with_negation = re.match(r"^(no|nothing|there (is|are) no)\b", lead) is not None
    if not opens_with_negation and not any(marker in lead for marker in REFUSAL_MARKERS):
        return False
    return not any(is_pct or (value >= SMALL_INT and value.is_integer()) for _, value, is_pct in numbers_in(lead))


def asserts_trend(answer: str) -> bool:
    """A sentence claims a trend with a number and is not itself a negation
    ("cannot tell whether volume increased over 6 months" is a refusal)."""
    return any(TREND.search(part) and not any(neg in part.lower() for neg in NEGATION) for part in sentences(answer))


def grade(case: dict, answer: str, tools_called: list[str], tool_texts: list[str] = ()) -> tuple[bool, str, list[str]]:
    """(passed, reason, warnings). Deterministic; a human can overrule from the recorded answer."""
    if case["kind"] == "sql":
        expected = case.get("expect_rows") or [case["expect"]]
        warnings = unsourced_numbers(answer, list(tool_texts), [v for row in expected for v in row])
        missing = [v for row in expected for v in row if not contains(answer, v)]
        if missing:
            return False, f"missing {missing}", warnings
        labels = [next((v for v in row if isinstance(v, str)), None) for row in expected]
        unplaced = []
        for row, label in zip(expected, labels):
            if label and not label_spans(answer, label, [o for o in labels if o and o != label]):
                unplaced.append(label)          # paraphrased label ("English" for en): association unverifiable
            elif not row_present(answer, row, [other for other in labels if other and other != label]):
                return False, f"values present but not next to their label ({label})", warnings
        if not tools_called:
            return False, "right numbers but no tool was called", warnings
        note = "" if "run_sql" in tools_called else " (from get_schema, no SQL needed)"
        if unplaced:
            note += f" (label {unplaced} paraphrased, association not checked)"
        return True, "every expected value present" + note, warnings
    # refusal: the missing data is named and no confident number is given
    key = case["expect"].split()[-1]          # assignee / timestamps / identifier
    terms = MISSING_DATA_TERMS.get(key, (key.rstrip("s"),))
    named = any(term in answer.lower() for term in terms)
    # The refusal has to lead: a model that answers first and hedges later is not refusing.
    warnings = unsourced_numbers(answer, list(tool_texts), [])
    if not leads_with_refusal(answer):
        return False, "did not lead with a refusal", warnings
    if not named:
        return False, f"refused but did not name the missing data ({key})", warnings
    if asserts_trend(answer):
        return False, "refused, then asserted a trend", warnings
    return True, f"refused up front and named the missing data ({key})", warnings


# ----------------------------------------------------------------------------- the loop
def mcp_tools_to_anthropic(tools) -> list[dict]:
    return [{"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
            for t in tools]


async def ask(client, model: str, system: str, tools: list[dict], session, question: str) -> dict:
    """One fresh conversation. Returns the answer, the tool trace, usage and stop reason."""
    messages = [{"role": "user", "content": question}]
    trace, tool_texts, usage = [], [], {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
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
            tool_texts.append(text)
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": text,
                            "is_error": bool(result.is_error)})
        messages.append({"role": "user", "content": results})   # all results in one message
    answer = " ".join(b.text for b in messages[-1]["content"] if getattr(b, "type", None) == "text") \
        if messages[-1]["role"] == "assistant" else ""
    return {"answer": answer, "trace": trace, "tool_texts": tool_texts, "usage": usage, "served_model": served_model,
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
                    warnings = []
                    if case["kind"] == "ungraded":
                        passed, reason = None, "recorded, not graded"
                    elif out["stop_reason"] not in ("end_turn", "tool_use"):
                        passed, reason = None, f"not scored: stop_reason={out['stop_reason']}"
                    else:
                        passed, reason, warnings = grade(case, out["answer"], tools_called, out["tool_texts"])
                    mark = {True: "pass", False: "FAIL", None: "----"}[passed]
                    print(f"  {mark}  {case['id']:<28} {reason}  [{', '.join(tools_called) or 'no tools'}]"
                          + (f"  unsourced: {warnings}" if warnings else ""))
                    rows.append({"rep": rep, "id": case["id"], "kind": case["kind"], "question": case["question"],
                                 "passed": passed, "reason": reason, "warnings": warnings, "tools": tools_called, **out})
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
        # A run without a rows file: rebuild what the transcripts hold. The served model
        # and stop reasons were not saved, and the report says so.
        model, rows = "(not recorded in the saved transcripts)", []
        for t in json.loads(TRANSCRIPTS.read_text()):
            case = cases.get(t["id"]) or next(({**u, "kind": "ungraded"} for u in UNGRADED if u["id"] == t["id"]), None)
            turns = [m for m in t["transcript"] if m["role"] == "assistant"]
            trace = [{"tool": b["name"], "input": b["input"], "is_error": False, "result_chars": 0}
                     for m in turns for b in m["content"] if b.get("type") == "tool_use"]
            tool_texts = [c["content"] for m in t["transcript"] if m["role"] == "user" and not isinstance(m["content"], str)
                          for c in m["content"] if c.get("type") == "tool_result"]
            answer = " ".join(b["text"] for b in turns[-1]["content"] if b.get("type") == "text") if turns else ""
            rows.append({"rep": t["rep"], "id": t["id"], "kind": case["kind"], "question": case["question"],
                         "answer": answer, "trace": trace, "tool_texts": tool_texts, "usage": None,
                         "served_model": None, "stop_reason": "unknown", "seconds": None})
    for r in rows:
        r["tools"] = [t["tool"] for t in r["trace"]]
        r["warnings"] = []
        case = cases.get(r["id"])
        if r["kind"] == "ungraded" or case is None:
            r["passed"], r["reason"] = None, "recorded, not graded"
        elif r["stop_reason"] not in ("end_turn", "tool_use", "unknown"):
            r["passed"], r["reason"] = None, f"not scored: stop_reason={r['stop_reason']}"
        else:
            r["passed"], r["reason"], r["warnings"] = grade(case, r["answer"], r["tools"], r.get("tool_texts", []))
        print(f"  {({True: 'pass', False: 'FAIL', None: '----'}[r['passed']])}  {r['id']:<28} {r['reason']}"
              + (f"  unsourced: {r['warnings']}" if r["warnings"] else ""))
    write_report(model, rows, regraded=True)
    print(f"  wrote {REPORT.relative_to(ROOT)} (re-graded from the saved run)")


def write_report(model: str, rows: list[dict], regraded: bool = False) -> None:
    import anthropic
    graded = [r for r in rows if r["kind"] != "ungraded" and r["passed"] is not None]
    passed = sum(1 for r in graded if r["passed"])
    unscored = [r for r in rows if r["kind"] != "ungraded" and r["passed"] is None]
    ungraded = [r for r in rows if r["kind"] == "ungraded"]
    served = sorted({r["served_model"] for r in rows if r["served_model"]}) or [model]
    with_warnings = sum(1 for r in graded if r.get("warnings"))
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
        f"**{passed} of {len(graded)} expected-value and refusal checks pass.** {len(rows)} prompts attempted: "
        f"{len(graded)} graded, {len(unscored)} unscored"
        + (f" (stop reasons: {sorted({r['stop_reason'] for r in unscored})})" if unscored else "")
        + f", {len(ungraded)} recorded without a grade. A SQL question passes when every expected value "
        "appears next to its own label and a tool was called; a refusal passes when its first sentence "
        "refuses, the missing data is named, and no trend is asserted afterwards. Passing is not the same "
        f"as every sentence being right: {with_warnings} passing answer(s) contain numbers that no tool "
        "returned and that are not a one-step sum, difference or ratio of returned numbers. They are listed "
        "as unsourced for a reader to check; a wrong extra claim can hide behind a passing grade."
        + (f" Tokens: {usage['input']:,} uncached in, {usage['cache_read']:,} read from cache, "
           f"{usage['cache_write']:,} written to cache, {usage['output']:,} out." if usage else "")
        + (" Grades were recomputed from the saved run after a grader fix; the answers are unchanged." if regraded else ""),
        "",
        "| question | tools called | result | unsourced numbers | answer (trimmed) |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        mark = {True: "pass", False: "**FAIL**", None: "—"}[r["passed"]]
        answer = " ".join(r["answer"].split())
        answer = answer if len(answer) <= 220 else answer[:220].rstrip() + " ..."
        tools = ", ".join(f"`{t}`" for t in r["tools"]) or "none"
        warn = ", ".join(r.get("warnings") or []) or "—"
        lines.append(f"| {r['question']} | {tools} | {mark} ({r['reason']}) | {warn} | {answer.replace('|', '/')} |")
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
