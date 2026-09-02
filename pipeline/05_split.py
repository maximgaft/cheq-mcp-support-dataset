"""Stage 05 - assign each ticket to train / val / test.

Three splits, not two. `val` is where the similarity floor gets calibrated and k
gets chosen; `test` is read once, at the end, for the numbers we report. Sweeping
a threshold on the set you then report on is the same leak as training on
`Customer Status` - it just looks like a good result instead of a bug.

Stratified on queue x language. Queue because it is the routing target and the
classes run 21:1 (Technical Support 11.7k, General Inquiry 559). Language because
every retrieval metric gets sliced en/de, and an unstratified split can leave a
slice too thin to read.

Only train rows ever enter the embedding index. Test tickets are queries against
an index that has never seen them.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
IN = DATA / "interim" / "04_deduped.parquet"
OUT = DATA / "interim" / "05_split.parquet"

TEST_FRAC = 0.10
VAL_FRAC = 0.10
SEED = 42


def main() -> None:
    corpus = pd.read_parquet(IN)
    rng = np.random.default_rng(SEED)

    splits = pd.Series(index=corpus.index, dtype=object)
    for _, group in corpus.groupby(["queue", "language"], sort=False):
        order = rng.permutation(group.index.to_numpy())
        n_test = int(round(len(group) * TEST_FRAC))
        n_val = int(round(len(group) * VAL_FRAC))
        splits.loc[order[:n_test]] = "test"
        splits.loc[order[n_test : n_test + n_val]] = "val"
        splits.loc[order[n_test + n_val :]] = "train"

    corpus["split"] = splits
    assert corpus.split.notna().all(), "some rows were not assigned a split"

    print("  rows per split:")
    for split in ("train", "val", "test"):
        s = corpus[corpus.split.eq(split)]
        print(f"    {split:<6} {len(s):>6}  indexable {s.is_indexable.sum():>6}"
              f"  {s.language.value_counts(normalize=True).round(2).to_dict()}")

    print(f"\n  index will hold {corpus.query('split == \"train\" and is_indexable').shape[0]:,} tickets")

    print("\n  queue share per split (stratification check):")
    share = pd.crosstab(corpus.queue, corpus.split, normalize="columns").round(3)
    print(share.sort_values("train", ascending=False).to_string())

    corpus.to_parquet(OUT, index=False)
    print(f"\n  wrote {OUT.relative_to(DATA.parent)}")


if __name__ == "__main__":
    main()
