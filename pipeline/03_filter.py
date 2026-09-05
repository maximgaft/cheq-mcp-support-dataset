"""Stage 03 - split off the foreign source, drop corrupt records, flag index
eligibility, and re-derive the language label.

Three different decisions, kept apart on purpose:

  DROP  - the record is not a valid ticket here: empty body, a body that holds
          an agent's reply rather than the customer's message, or a body in a
          third language (a dozen es/fr/pt tickets inside a two-language corpus).
  FLAG  - the ticket is real but cannot serve as retrieval precedent: no answer,
          a body or answer under 5 words, or an answer in a different language
          from the body. It still counts for every aggregate, so run_sql counts
          it and the embedding index does not.
  RELABEL - `language` is re-derived from the body. The source label says `de`
          on about 5,300 rows whose text is English (4,204 after dedup) -
          10% of the corpus, a handful the other way - so stratifying or slicing
          on it would be 25% wrong for German. The original is kept as
          `language_label` so the defect stays measurable in SQL. Detection is a
          function-word count over en/de/fr/es/pt. A body that does not decide it
          defers to the answer, then to the source label; a third language needs
          3 hits before it counts.

german_norm goes to its own file rather than the corpus. It is a different
dataset that shares five column names (subject, body, queue, priority, language)
and nothing else: no answers, no type, no tags,
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
MIN_FOREIGN_HITS = 3  # a third language needs this many function words before it counts

WORD = re.compile(r"[a-zA-ZäöüßÄÖÜ]{4,}")
AGENT_GREETING = r"^\s*(dear|hi|hello|guten tag|hallo|sehr geehrte[r]?)\s*<name>"

# Function words per language, excluding any that are common in another listed
# language: no "in"/"an"/"am" (en and de), no "des" (fr and de), no "as"/"com" (pt and en).
LANGUAGE_WORDS = {
    "en": "the and to is are for with our your please have has we you that this from not of a my i "
          "be by can which would should been there their they was were or as on at it its any all",
    "de": "der die das und ist sind für mit unser ihr ihre bitte haben hat wir sie dass nicht ich eine "
          "einen zu ein auf im bei von vom zum zur über aus nach oder auch noch wird werden wurde kann "
          "können muss müssen es sich wenn als aber dem den des einer eines um sehr hier diese dieser "
          "dieses keine kein ob mein meine unsere unserer ihnen ihrer ihrem",
    "fr": "le la les une est pour avec nous vous dans pas sur votre notre je cette ce",
    "es": "el los las una para con usted que por su nuestro del está tenemos necesito",
    "pt": "os uma é para você nós que por sua nosso não está temos preciso",
}
LANGUAGE_WORDS = {
    lang: re.compile(r"\b(?:" + "|".join(words.split()) + r")\b", re.I)
    for lang, words in LANGUAGE_WORDS.items()
}


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


def detect_language(text: str) -> str | None:
    """"en", "de", "other" for a third language, or None when the words do not decide it."""
    hits = {lang: len(rx.findall(text)) for lang, rx in LANGUAGE_WORDS.items()}
    foreign = max(hits["fr"], hits["es"], hits["pt"])
    scores = {"en": hits["en"], "de": hits["de"],
              "other": foreign if foreign >= MIN_FOREIGN_HITS else 0}
    best = max(scores, key=scores.get)
    if scores[best] == 0 or list(scores.values()).count(scores[best]) > 1:
        return None
    return best


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

    # Language: the body decides; a body too short to decide defers to the answer;
    # only then does the source label stand. That label is wrong on a quarter of
    # its `de` rows, so it is the last resort rather than the default.
    body_language = corpus.body.map(detect_language)
    answer_language = corpus.answer.map(lambda a: detect_language(a) if isinstance(a, str) else None)
    language = body_language.fillna(answer_language).fillna(corpus.language)
    foreign = language.eq("other")
    print(f"  dropped: other language {foreign.sum():>3}  (es/fr/pt in a two-language corpus)")
    keep = ~foreign
    corpus, body_language, answer_language, language = (
        corpus[keep].copy(), body_language[keep], answer_language[keep], language[keep])

    corpus["language_label"] = corpus.language
    corpus["language"] = language
    relabelled = corpus.language.ne(corpus.language_label)
    print(f"\n  language relabelled  {relabelled.sum():>6}  ({100 * relabelled.mean():.1f}%)")
    for (was, now), n in corpus[relabelled].groupby(["language_label", "language"]).size().items():
        print(f"    {was} -> {now} {n:>6}")

    corpus["body_words"] = corpus.body.str.split().str.len()
    corpus["answer_words"] = corpus.answer.fillna("").str.split().str.len()
    has_answer = corpus.answer_words > 0
    body_long = corpus.body_words >= MIN_INDEX_WORDS
    answer_long = corpus.answer_words >= MIN_INDEX_WORDS
    # A mismatch needs both sides decided: an English body answered in German is
    # not a usable precedent, but an undecidable body is not evidence of anything.
    same_language = body_language.isna() | answer_language.isna() | body_language.eq(answer_language)
    corpus["is_indexable"] = body_long & answer_long & same_language

    print(f"\n  corpus kept          {len(corpus):>6}")
    print(f"    indexable          {corpus.is_indexable.sum():>6}"
          f"  ({100 * corpus.is_indexable.mean():.1f}%)")
    print(f"    not: no answer               {(~has_answer).sum():>6}")
    print(f"    not: body < {MIN_INDEX_WORDS} words          {(~body_long).sum():>6}")
    print(f"    not: answer < {MIN_INDEX_WORDS} words        {(has_answer & ~answer_long).sum():>6}")
    print(f"    not: answer in other language {(~same_language).sum():>6}")

    corpus.to_parquet(OUT_CORPUS, index=False)
    print(f"\n  wrote {OUT_CORPUS.relative_to(DATA.parent)}")


if __name__ == "__main__":
    main()
