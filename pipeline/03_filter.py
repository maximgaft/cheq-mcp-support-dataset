"""Stage 03 - split off the foreign source, drop corrupt records, flag index eligibility.

Two different decisions, kept apart on purpose:

  DROP  - the record is not a valid ticket (empty body, or the body holds an
          agent's reply rather than the customer's message).
  FLAG  - the ticket is real but cannot serve as retrieval precedent (no answer
          to retrieve, or too short to carry signal). It still counts for every
          aggregate, so run_sql counts it and the embedding index does not.

german_norm goes to its own file rather than the corpus. It is a different
dataset that happens to share three column names: no answers, no type, no tags,
a 42-value queue taxonomy instead of 10, a 5-value priority scale instead of 3,
and 100% German. It can join neither the analytical nor the precedent story.

It is dropped, not written out. An earlier version kept it as an abstention test
set; that was wrong - those tickets are semantically in-domain, so retrieval finds
plausible neighbours for them, correctly. Abstention is measured against
handwritten queries in evals/abstention_queries.yaml instead, and nothing else
ever read the file.
"""

import re
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
IN = DATA / "interim" / "02_cleaned.parquet"
OUT_CORPUS = DATA / "interim" / "03_corpus.parquet"

MIN_INDEX_WORDS = 5  # 1-4 words is "Requesting Assistance"; 5+ carries a topic
ROLE_OVERLAP = 0.8

WORD = re.compile(r"[a-zA-ZäöüßÄÖÜ]{4,}")
AGENT_GREETING = r"^\s*(dear|hi|hello|guten tag|hallo|sehr geehrte[r]?)\s*<name>"


def role_confused(df: pd.DataFrame) -> pd.Series:
    """Body holds support-side text: it greets the customer by name, or it is a
    near-copy of the answer field."""
    greets = df.body.str.match(AGENT_GREETING, case=False, na=False)
    pairs = zip(df.body.fillna(""), df.answer.fillna(""))
    overlap = pd.Series(
        [
            len(set(WORD.findall(b.lower())) & set(WORD.findall(a.lower())))
            / max(len(set(WORD.findall(b.lower())) | set(WORD.findall(a.lower()))), 1)
            for b, a in pairs
        ],
        index=df.index,
    )
    return greets | (overlap > ROLE_OVERLAP)


def main() -> None:
    tickets = pd.read_parquet(IN)

    german = tickets[tickets.source.eq("german_norm")]
    corpus = tickets[tickets.source.ne("german_norm")].copy()
    print(f"  german_norm split off {len(german):>5}")
    print(f"  corpus               {len(corpus):>6}")

    no_body = corpus.body.isna() | corpus.body.str.strip().eq("")
    confused = role_confused(corpus)
    print(f"\n  dropped: empty body  {no_body.sum():>6}")
    print(f"  dropped: role-confused {confused.sum():>4}")
    corpus = corpus[~(no_body | confused)].copy()

    corpus["body_words"] = corpus.body.str.split().str.len()
    has_answer = corpus.answer.notna() & corpus.answer.str.strip().ne("")
    long_enough = corpus.body_words >= MIN_INDEX_WORDS
    corpus["is_indexable"] = has_answer & long_enough

    print(f"\n  corpus kept          {len(corpus):>6}")
    print(f"    indexable          {corpus.is_indexable.sum():>6}"
          f"  ({100 * corpus.is_indexable.mean():.1f}%)")
    print(f"    not: no answer     {(~has_answer).sum():>6}")
    print(f"    not: < {MIN_INDEX_WORDS} words     {(~long_enough).sum():>6}")

    corpus.to_parquet(OUT_CORPUS, index=False)
    print(f"\n  wrote {OUT_CORPUS.relative_to(DATA.parent)}")


if __name__ == "__main__":
    main()
