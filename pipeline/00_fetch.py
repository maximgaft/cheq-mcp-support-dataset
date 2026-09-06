"""Stage 00 - download the three source CSVs and verify their checksums.

Python rather than a shell script so the build has one runtime: no bash, no
`shasum`, so it runs the same on a Windows laptop and a slim Linux image.

The dataset ships as three files with different schemas; stage 01 tags each row
with the file it came from, because which columns a row has depends on its source.
A file that is already present and matches its checksum is not fetched again.
"""

import hashlib
import os
import ssl
import sys
import urllib.request
from pathlib import Path

import certifi

DATA = Path(__file__).resolve().parents[1] / "data"
REPO = "https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets/resolve/main"

CHECKSUMS = {
    "aa_dataset-tickets-multi-lang-5-2-50-version.csv":
        "f187c090e59581c2bbf3aa1377c8db4dd647464ecf2ae51bf8966e42e0ed6bc0",
    "dataset-tickets-german_normalized_50_5_2.csv":
        "22580337aed864d8c0485f16a7fe683d48d8adfc3af0cd1a6fe1e240f728735f",
    "dataset-tickets-multi-lang-4-20k.csv":
        "9be3bf810584fe01e8e83383e83dfd33f4c3910938ecad03ef151da79d8f0635",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ca_bundle() -> str:
    """The root certificates the download is verified against.

    A python.org build with no certificates installed trusts nothing and fails every
    HTTPS fetch, so certifi's Mozilla bundle is the default rather than the interpreter's.
    SSL_CERT_FILE wins when set: that is the rule httpx applies to the model download,
    so one variable covers both fetches behind a TLS-inspecting proxy whose root the
    Mozilla bundle does not carry.
    """
    return os.environ.get("SSL_CERT_FILE") or certifi.where()


def download(url: str, path: Path) -> None:
    # urllib follows Hugging Face's 307 redirect to the resolve-cache URL.
    context = ssl.create_default_context(cafile=ca_bundle())
    with urllib.request.urlopen(url, context=context) as response, path.open("wb") as out:
        for chunk in iter(lambda: response.read(1 << 20), b""):
            out.write(chunk)


def main() -> None:
    DATA.mkdir(exist_ok=True)
    for name, want in CHECKSUMS.items():
        path = DATA / name
        if path.exists() and sha256(path) == want:
            print(f"  ok       {name}")
            continue
        print(f"  fetching {name}")
        try:
            download(f"{REPO}/{name}", path)
        except OSError as exc:  # URLError and HTTPError are OSErrors
            path.unlink(missing_ok=True)
            sys.exit(f"  DOWNLOAD FAILED for {name}: {exc}")
        got = sha256(path)
        if got != want:
            path.unlink(missing_ok=True)
            sys.exit(f"  CHECKSUM MISMATCH for {name}: got {got[:16]}..., wanted {want[:16]}...")
    print("  all three verified")


if __name__ == "__main__":
    main()
