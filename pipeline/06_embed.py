"""Stage 06 - embed the indexable training tickets into the retrieval index.

Only `split == "train" and is_indexable` goes in. Val and test tickets are
queries against an index that has never seen them, which is what makes the
retrieval numbers mean anything.

Vectors are centred (see embedding.center) and the mean is stored alongside
them, because the server has to apply the identical transform to queries.

Ends with two checks. The random-pair number confirms centring worked - it
should sit near zero. The nearest-neighbour sample confirms the index retrieves
related tickets at all; if it doesn't, nothing built on top will work, and it is
far cheaper to learn that here than after five tools are written.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.embedding import (  # noqa: E402
    DIM,
    MODEL_NAME,
    center,
    document_text,
    embed_documents,
    embed_query,
)

DATA = Path(__file__).resolve().parents[1] / "data"
IN = DATA / "interim" / "05_split.parquet"
OUT = DATA / "interim" / "06_index.npz"


def main() -> None:
    corpus = pd.read_parquet(IN)
    train = corpus[corpus.split.eq("train") & corpus.is_indexable].reset_index(drop=True)
    texts = [document_text(s, b) for s, b in zip(train.subject, train.body)]

    print(f"  model {MODEL_NAME} ({DIM}d)")
    print(f"  embedding {len(texts):,} tickets...")
    raw = embed_documents(texts, progress=True)

    mean = raw.mean(axis=0)
    vectors = center(raw, mean)
    assert vectors.shape == (len(train), DIM), f"unexpected shape {vectors.shape}"

    rng = np.random.default_rng(0)
    i, j = rng.integers(0, len(vectors), 5000), rng.integers(0, len(vectors), 5000)
    print(f"\n  unrelated-pair similarity: raw {(raw[i] * raw[j]).sum(1).mean():+.3f}"
          f"  ->  centred {(vectors[i] * vectors[j]).sum(1).mean():+.3f}   (want ~0)")

    # ticket_id as a fixed-width string array, not a pandas object array, so the
    # file loads without allow_pickle.
    np.savez_compressed(OUT, ticket_id=np.asarray(train.ticket_id, dtype=str),
                        vectors=vectors, mean=mean)
    print(f"  {vectors.nbytes / 1e6:.1f} MB in memory, wrote {OUT.relative_to(DATA.parent)}")

    print("\n  nearest-neighbour spot check (3 held-out queries):")
    for _, row in corpus[corpus.split.eq("test")].sample(3, random_state=7).iterrows():
        q = center(embed_query(document_text(row.subject, row.body)), mean)
        sims = vectors @ q
        print(f"\n    QUERY [{row.queue} / {row.language}] {str(row.subject)[:58]!r}")
        for rank, k in enumerate(np.argsort(-sims)[:3], 1):
            hit = train.iloc[k]
            print(f"      {rank}. {sims[k]:.3f}  [{hit.queue} / {hit.language}] {str(hit.subject)[:52]!r}")


if __name__ == "__main__":
    main()
