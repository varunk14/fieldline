"""Download the manual corpus from Carrier's public document host.

The PDFs and the text extracted from them are not redistributed in this
repository - they are Carrier's copyrighted documentation, published openly by
Carrier but not ours to republish. This script fetches them so a clean checkout
can still rebuild the index from nothing.

Usage:
    python scripts/fetch_corpus.py
    python scripts/fetch_corpus.py --force
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "corpus" / "raw"

TIMEOUT_SECONDS = 120
USER_AGENT = "fieldline-corpus-fetch/0.1"


@dataclass(frozen=True)
class Manual:
    """One published document, with the provenance the README cites."""

    filename: str
    url: str
    title: str


# Carrier WeatherMaker 48TC, 17-30 ton. One model family, as PRD section 2
# requires. All three are served from Carrier's own shareddocs host rather than
# a manual-aggregator site, so the provenance is unambiguous.
MANUALS = (
    Manual(
        filename="48TC-17-30-02SM.pdf",
        url="https://www.shareddocs.com/hvac/docs/1005/Public/01/48TC-17-30-02SM.pdf",
        title="48TC 17-30 Service and Maintenance Instructions",
    ),
    Manual(
        filename="48TC-17-30-V-06SI.pdf",
        url="https://www.shareddocs.com/hvac/docs/1005/Public/0D/48TC-17-30-V-06SI.pdf",
        title="48TC 17-30 Installation Instructions",
    ),
    Manual(
        filename="48TC-17-30-V-06PD.pdf",
        url="https://www.shareddocs.com/hvac/docs/1005/Public/02/48TC-17-30-V-06PD.pdf",
        title="48TC 17-30 Product Data",
    ),
)


def fetch(manual: Manual, target: Path, force: bool) -> bool:
    """Download one manual. Returns True if a download happened."""
    destination = target / manual.filename
    if destination.exists() and not force:
        print(f"  have    {manual.filename}")
        return False

    print(f"  fetch   {manual.filename} ... ", end="", flush=True)
    request = urllib.request.Request(manual.url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            payload = response.read()
    except (urllib.error.URLError, TimeoutError) as error:
        print("failed")
        raise SystemExit(f"Could not fetch {manual.url}: {error}") from error

    if not payload.startswith(b"%PDF"):
        raise SystemExit(f"{manual.url} did not return a PDF. Has the document moved?")

    destination.write_bytes(payload)
    print(f"{len(payload) // 1024} KB")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch the RTU manual corpus.")
    parser.add_argument("--dir", type=Path, default=RAW_DIR)
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    args = parser.parse_args(argv)

    args.dir.mkdir(parents=True, exist_ok=True)
    print(f"Corpus: Carrier WeatherMaker 48TC, 17-30 ton -> {args.dir}")

    for manual in MANUALS:
        fetch(manual, args.dir, args.force)

    print()
    print("Next: python scripts/ingest.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
