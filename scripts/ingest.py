"""Extract and chunk the manual corpus to JSON.

This script never contacts Moss. Chunking is tuned by re-running it, and index
builds are a separate, deliberate step - see scripts/build_index.py. Keeping
the tuning loop entirely offline means iterating on chunk quality cannot spend
index credits.

Usage:
    python scripts/ingest.py
    python scripts/ingest.py --raw corpus/raw --out corpus/chunks.json
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from collections import Counter
from pathlib import Path

from fieldline.corpus.chunk import chunk_manual
from fieldline.corpus.models import Chunk

logger = logging.getLogger("ingest")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = REPO_ROOT / "corpus" / "raw"
DEFAULT_OUT = REPO_ROOT / "corpus" / "chunks.json"

# Model family is a metadata field and a filter, so it is declared rather than
# guessed from the filename.
MODEL_FAMILY = "48TC"


def manual_id_for(pdf_path: Path) -> str:
    """Stable identifier derived from the published document number."""
    return pdf_path.stem.upper()


def build(raw_dir: Path) -> list[Chunk]:
    pdfs = sorted(raw_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {raw_dir}")

    chunks: list[Chunk] = []
    for pdf_path in pdfs:
        manual_id = manual_id_for(pdf_path)
        logger.info("Chunking %s", pdf_path.name)
        chunks.extend(chunk_manual(pdf_path, manual_id, MODEL_FAMILY))
    return chunks


def report(chunks: list[Chunk]) -> None:
    """Print the statistics the MVP 1 definition of done is written against."""
    if not chunks:
        print("No chunks produced.")
        return

    lengths = [c.word_count for c in chunks]
    by_type = Counter(c.content_type.value for c in chunks)
    by_manual = Counter(c.manual_id for c in chunks)

    print()
    print(f"chunks            {len(chunks)}")
    print(f"mean words        {statistics.mean(lengths):.1f}")
    print(f"median words      {statistics.median(lengths):.0f}")
    print(f"stdev words       {statistics.pstdev(lengths):.1f}")
    print(f"min / max words   {min(lengths)} / {max(lengths)}")
    print(f"distinct sections {len({c.section for c in chunks})}")
    print()
    print("by content type")
    for name, count in by_type.most_common():
        print(f"  {name:<12} {count}")
    print()
    print("by manual")
    for name, count in by_manual.most_common():
        print(f"  {name:<28} {count}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Chunk RTU manuals to JSON.")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    chunks = build(args.raw)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = [c.to_dict() for c in chunks]
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    report(chunks)
    print()
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
