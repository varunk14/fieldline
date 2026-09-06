"""Query the Moss index from the command line.

The tool for eyeballing retrieval quality by hand. Prints scores and the
reported `time_taken_ms` so latency is visible on every call rather than
measured only in a benchmark.

Usage:
    python scripts/query.py "unit is showing 7 flashes"
    python scripts/query.py "what torque for the fan bolts" --top-k 5
    python scripts/query.py "high pressure" --type fault_code
    python scripts/query.py --file eval/hand_queries.txt
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import statistics
import sys
from pathlib import Path

from fieldline.config import Settings
from fieldline.retrieval import DEFAULT_ALPHA, DEFAULT_TOP_K, MossRetriever, Retrieval

logger = logging.getLogger("query")

RULE = "-" * 78


def render(result: Retrieval, show_text: bool) -> None:
    print(RULE)
    print(f"query   {result.query}")
    print(f"latency {result.time_taken_ms:.2f}ms   hits {len(result.docs)}")
    print(RULE)
    for rank, doc in enumerate(result.docs, 1):
        header = f"{rank}. [{doc.score:.3f}] {doc.manual_id} p{doc.page}"
        print(f"{header}  {doc.section}")
        if show_text:
            body = doc.text.split("\n", 1)[-1].strip()
            print(f"     {body[:300]}{'...' if len(body) > 300 else ''}")
    print()


async def run(args: argparse.Namespace) -> int:
    settings = Settings.from_env()
    retriever = MossRetriever(
        settings.moss_project_id,
        settings.moss_project_key,
        settings.moss_index_name,
    )
    await retriever.load()

    if args.file:
        queries = [
            line.strip()
            for line in Path(args.file).read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    else:
        queries = args.query

    if not queries:
        raise SystemExit("Nothing to query. Pass a query or --file.")

    latencies: list[float] = []
    for text in queries:
        result = await retriever.aquery(
            text, top_k=args.top_k, alpha=args.alpha, content_type=args.type
        )
        latencies.append(result.time_taken_ms)
        render(result, show_text=not args.brief)

    if len(latencies) > 1:
        ordered = sorted(latencies)
        p95 = ordered[max(0, int(len(ordered) * 0.95) - 1)]
        print(f"{len(latencies)} queries   "
              f"mean {statistics.mean(latencies):.2f}ms   "
              f"median {statistics.median(latencies):.2f}ms   "
              f"p95 {p95:.2f}ms   max {max(latencies):.2f}ms")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Query the Moss index.")
    parser.add_argument("query", nargs="*", help="one or more queries")
    parser.add_argument("--file", type=Path, help="read queries from a file, one per line")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--type", help="filter on content_type")
    parser.add_argument("--brief", action="store_true", help="scores only, no chunk text")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
