#!/usr/bin/env bash
# Download the three source CSVs from Hugging Face and verify their checksums.
#
# The dataset ships as three separate files with different schemas; the pipeline
# tags each row with the file it came from, because which columns a row has
# depends on its source. See pipeline/01_load.py.
set -euo pipefail
cd "$(dirname "$0")"

REPO="https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets/resolve/main"

verify() {
  echo "$1  $2" | shasum -a 256 --check --status
}

fetch() {
  local file="$1" want="$2"
  if [ -f "$file" ] && verify "$want" "$file"; then
    echo "  ok       $file"
    return
  fi
  echo "  fetching $file"
  curl -sSLf -o "$file" "$REPO/$file"
  verify "$want" "$file" || { echo "  CHECKSUM MISMATCH for $file" >&2; exit 1; }
}

fetch aa_dataset-tickets-multi-lang-5-2-50-version.csv f187c090e59581c2bbf3aa1377c8db4dd647464ecf2ae51bf8966e42e0ed6bc0
fetch dataset-tickets-german_normalized_50_5_2.csv     22580337aed864d8c0485f16a7fe683d48d8adfc3af0cd1a6fe1e240f728735f
fetch dataset-tickets-multi-lang-4-20k.csv             9be3bf810584fe01e8e83383e83dfd33f4c3910938ecad03ef151da79d8f0635
echo "  all three verified"
