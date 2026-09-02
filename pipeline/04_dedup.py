"""Stage 04 - collapse tickets that appear in more than one source file.

Every duplicate group spans multilang_v1 and multilang_v2, so these are the same
tickets shipped in two dataset releases rather than distinct tickets that happen
to share wording. That makes them safe - and necessary - to drop: counting one
ticket twice would inflate every aggregate.

This must run before the train/test split. Left in, roughly a quarter of the test
set would have its twin sitting in training, and both the routing classifier and
the retrieval eval would score memorisation as skill.

We keep the first occurrence, which is the multilang_v1 copy - the only file
carrying the `version` column, so it retains the most metadata.
"""

from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
IN = DATA / "interim" / "03_corpus.parquet"
OUT = DATA / "interim" / "04_deduped.parquet"


def main() -> None:
    corpus = pd.read_parquet(IN)
    before = len(corpus)

    counts = corpus.body.value_counts()
    groups = counts[counts > 1]
    variant_answers = int(
        (corpus[corpus.body.isin(groups.index)].groupby("body").answer.nunique(dropna=False) > 1).sum()
    )

    print(f"  in                    {before:>6}")
    print(f"  duplicate groups      {len(groups):>6}  (copies each: {sorted(groups.unique())})")
    print(f"  ...of which carry a differing answer: {variant_answers}")

    deduped = corpus.drop_duplicates(subset="body", keep="first").copy()
    print(f"\n  removed               {before - len(deduped):>6}")
    print(f"  out                   {len(deduped):>6}")
    print(f"    indexable           {deduped.is_indexable.sum():>6}"
          f"  ({100 * deduped.is_indexable.mean():.1f}%)")
    print(f"    by language         {deduped.language.value_counts().to_dict()}")

    deduped.to_parquet(OUT, index=False)
    print(f"\n  wrote {OUT.relative_to(DATA.parent)}")


if __name__ == "__main__":
    main()
