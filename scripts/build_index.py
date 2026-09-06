"""Build the Moss index from corpus/chunks.json.

This is the only script that spends index credit, so it is deliberately
separate from ingestion and refuses to run by accident. Chunking is tuned by
re-running scripts/ingest.py, which never touches Moss.

Every call is written to the ledger with a timestamp, including failures - a
failed build may still have spent the credit.

Usage:
    python scripts/build_index.py                      # dry run, the default
    python scripts/build_index.py --confirm            # builds
    python scripts/build_index.py --confirm --replace  # rebuilds over an existing index
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from fieldline.config import DEFAULT_CHUNKS, Settings
from fieldline.corpus.models import Chunk
from fieldline.retrieval import MossRetriever
from fieldline.retrieval import ledger as ledger_module

logger = logging.getLogger("build_index")


def load_chunks(path: Path) -> list[Chunk]:
    if not path.exists():
        raise SystemExit(f"No chunks at {path}. Run scripts/ingest.py first.")
    return [Chunk.from_dict(d) for d in json.loads(path.read_text())]


def summarise(chunks: list[Chunk]) -> None:
    total_words = sum(c.word_count for c in chunks)
    print(f"  chunks      {len(chunks)}")
    print(f"  words       {total_words}")
    print(f"  manuals     {len({c.manual_id for c in chunks})}")
    print(f"  sections    {len({c.section for c in chunks})}")


def show_ledger() -> None:
    entries = ledger_module.read()
    if not entries:
        print("  (no prior ingest calls recorded)")
        return
    for entry in entries:
        print(
            f"  {entry.timestamp}  {entry.operation:<12} {entry.index_name:<20} "
            f"{entry.doc_count:>5} docs  {entry.duration_ms:>9.0f}ms  {entry.outcome}"
        )


async def run(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    chunks = load_chunks(args.chunks)

    print(f"Index      {settings.moss_index_name}")
    summarise(chunks)
    print()
    print("Prior ingest calls in the ledger:")
    show_ledger()
    print()

    if not args.confirm:
        print("Dry run. This would call create_index once, which spends tier credit.")
        print("Re-run with --confirm to build.")
        return 0

    retriever = MossRetriever(
        settings.moss_project_id,
        settings.moss_project_key,
        settings.moss_index_name,
    )

    if args.replace:
        # The Developer tier caps a project at three indexes, so a rebuild has
        # to free the name first rather than building alongside the old one.
        print(f"Deleting existing index {settings.moss_index_name}...")
        await retriever.delete_index()
        print("  deleted")

    print("Embedding chunks locally before upload...")
    print(f"Building index with {len(chunks)} documents. This blocks until the build finishes.")
    job_id = await retriever.create_index(chunks)
    print(f"  build complete (job {job_id or 'n/a'})")

    print("Loading index into this process...")
    await retriever.load()
    print("  loaded, and the probe query was served locally")

    result = await retriever.aquery("what does 7 flashes mean", top_k=3)
    print()
    print(f"Smoke query returned {len(result.docs)} docs in {result.time_taken_ms:.1f}ms")
    for doc in result.docs:
        print(f"  [{doc.score:.3f}] p{doc.page} {doc.section[:48]}")

    print()
    print("Ledger now:")
    show_ledger()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Moss index.")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="actually build; without this the script only reports what it would do",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="delete the existing index first; required when rebuilding",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
