"""Stage 02 - normalise the text fields. No rows are added or removed.

Line breaks in this corpus are encoded two ways: as the literal two-character
sequence \\n (13k bodies) and as <br> tags (1.4k bodies). Both become real
newlines here.

Other angle-bracket tokens are left alone on purpose - <name>, <tel_num>,
<acc_num> and friends are the dataset's anonymisation placeholders, and a
strip-all-tags regex would delete meaning we want to keep.
"""

from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
IN = DATA / "interim" / "01_loaded.parquet"
OUT = DATA / "interim" / "02_cleaned.parquet"

TEXT_COLS = ["subject", "body", "answer"]
LITERAL_NEWLINE = chr(92) + "n"


def clean(s: pd.Series) -> pd.Series:
    """Nulls stay null - .str methods propagate NaN, which is what we want."""
    return (
        s.str.replace(LITERAL_NEWLINE, "\n", regex=False)
        .str.replace(r"<br\s*/?>", "\n", regex=True)
        .str.replace(r"\n{3,}", "\n\n", regex=True)
        .str.strip()
    )


def main() -> None:
    tickets = pd.read_parquet(IN)
    before = len(tickets)

    for col in TEXT_COLS:
        original = tickets[col]
        tickets[col] = clean(original)
        changed = (original.fillna("") != tickets[col].fillna("")).sum()
        print(f"  {col:<8} {changed:>6} of {original.notna().sum():>6} values changed")

    assert len(tickets) == before, f"row count changed: {before} -> {len(tickets)}"

    example = tickets.loc[tickets.body.fillna("").str.contains("\n"), "body"].iloc[0]
    print(f"\n  example cleaned body:\n    {example[:120]!r}")

    tickets.to_parquet(OUT, index=False)
    print(f"\n  {len(tickets):,} rows unchanged, wrote {OUT.relative_to(DATA.parent)}")


if __name__ == "__main__":
    main()
