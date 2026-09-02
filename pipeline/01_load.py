"""Stage 01 - load the three source CSVs into one table, tagged by origin.

The dataset ships as three CSVs with different schemas, so the columns a row
has depend on which file it came from. We tag each row with its source and let
later stages filter on that, rather than on row position.
"""

from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
OUT = DATA / "interim" / "01_loaded.parquet"

SOURCES = {
    "multilang_v1": "aa_dataset-tickets-multi-lang-5-2-50-version.csv",
    "german_norm": "dataset-tickets-german_normalized_50_5_2.csv",
    "multilang_v2": "dataset-tickets-multi-lang-4-20k.csv",
}

EXPECTED_ROWS = 61_765


def main() -> None:
    frames = []
    for source, filename in SOURCES.items():
        # dtype=str so pandas never guesses a type at this stage; later stages cast
        # deliberately. Otherwise `version` infers as int in one file and float in another.
        df = pd.read_csv(DATA / filename, dtype=str)
        df["source"] = source
        # The data ships with no identifier. Position within the source file is
        # stable across reruns and traces a row back to the CSV it came from.
        df["ticket_id"] = [f"{source}:{i}" for i in range(len(df))]
        print(f"  {source:<13} {len(df):>6} rows  {len(df.columns) - 1:>2} columns")
        frames.append(df)

    tickets = pd.concat(frames, ignore_index=True)
    assert len(tickets) == EXPECTED_ROWS, f"expected {EXPECTED_ROWS:,}, got {len(tickets):,}"

    print(f"\n  combined: {len(tickets):,} rows x {len(tickets.columns)} columns")
    print("\n  non-null values per column per source (0 = column absent in that file):")
    print(tickets.groupby("source").count().T.to_string())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tickets.to_parquet(OUT, index=False)
    print(f"\n  wrote {OUT.relative_to(DATA.parent)}")


if __name__ == "__main__":
    main()
