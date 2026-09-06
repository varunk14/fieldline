"""Read and audit chunks without an index existing.

Chunk quality is decided before anything reaches Moss, so it has to be
inspectable offline. This is the tool for reading chunks by hand, checking that
fault codes stayed with their descriptions, and finding the chunks that
statistics alone would never flag - a page of two-column text spliced into word
salad has a perfectly ordinary word count.

Usage:
    python scripts/inspect_chunks.py stats
    python scripts/inspect_chunks.py show --id 48TC-17-30-02SM-a1b2c3d4e5f6
    python scripts/inspect_chunks.py show --page 45
    python scripts/inspect_chunks.py grep "flash"
    python scripts/inspect_chunks.py sample --n 5 --type fault_code
    python scripts/inspect_chunks.py codes
    python scripts/inspect_chunks.py suspect
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

from fieldline.corpus.models import Chunk

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS = REPO_ROOT / "corpus" / "chunks.json"

# Codes a technician would actually read off a board or a display.
CODE_RE = re.compile(r"\b(\d{1,2}\s*flash(?:es)?|E-?\d{1,3}|A\d{2,3})\b", re.IGNORECASE)

RULE = "-" * 78


def load(path: Path) -> list[Chunk]:
    if not path.exists():
        raise SystemExit(f"No chunk file at {path}. Run scripts/ingest.py first.")
    return [Chunk.from_dict(d) for d in json.loads(path.read_text())]


def show_chunk(chunk: Chunk, full: bool = True) -> None:
    print(RULE)
    print(f"id       {chunk.id}")
    print(f"manual   {chunk.manual_id}   page {chunk.page}   words {chunk.word_count}")
    print(f"section  {chunk.section}")
    print(f"type     {chunk.content_type.value}   source {chunk.extra.get('source', '?')}")
    print(RULE)
    text = chunk.text if full else chunk.text[:400] + "..."
    print(text)
    print()


def cmd_stats(chunks: list[Chunk]) -> int:
    lengths = [c.word_count for c in chunks]
    print(f"chunks            {len(chunks)}")
    print(f"mean words        {statistics.mean(lengths):.1f}")
    print(f"median words      {statistics.median(lengths):.0f}")
    print(f"stdev words       {statistics.pstdev(lengths):.1f}")
    print(f"min / max words   {min(lengths)} / {max(lengths)}")
    print(f"under 20 words    {sum(1 for n in lengths if n < 20)}")
    print(f"over 400 words    {sum(1 for n in lengths if n > 400)}")
    print(f"distinct sections {len({c.section for c in chunks})}")
    print()
    for name, count in Counter(c.content_type.value for c in chunks).most_common():
        print(f"  {name:<12} {count}")
    return 0


def cmd_show(chunks: list[Chunk], args: argparse.Namespace) -> int:
    selected = chunks
    if args.id:
        selected = [c for c in selected if c.id == args.id]
    if args.page is not None:
        selected = [c for c in selected if c.page == args.page]
    if args.section:
        selected = [c for c in selected if args.section.lower() in c.section.lower()]
    if not selected:
        print("No matching chunks.")
        return 1
    for chunk in selected[: args.limit]:
        show_chunk(chunk)
    print(f"{len(selected)} matched, showing up to {args.limit}")
    return 0


def cmd_grep(chunks: list[Chunk], args: argparse.Namespace) -> int:
    pattern = re.compile(args.pattern, re.IGNORECASE)
    hits = [c for c in chunks if pattern.search(c.text)]
    for chunk in hits[: args.limit]:
        show_chunk(chunk, full=not args.brief)
    print(f"{len(hits)} chunks matched {args.pattern!r}")
    return 0 if hits else 1


def cmd_sample(chunks: list[Chunk], args: argparse.Namespace) -> int:
    pool = chunks
    if args.type:
        pool = [c for c in pool if c.content_type.value == args.type]
    if not pool:
        print("No chunks of that type.")
        return 1
    rng = random.Random(args.seed)
    for chunk in rng.sample(pool, min(args.n, len(pool))):
        show_chunk(chunk)
    return 0


def cmd_codes(chunks: list[Chunk], args: argparse.Namespace) -> int:
    """Check every fault code found in the corpus kept its description.

    This is the integrity check that matters most: a code severed from its
    meaning retrieves for the wrong thing and is the failure the whole
    chunking strategy exists to prevent.
    """
    found: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        for match in CODE_RE.findall(chunk.text):
            key = re.sub(r"\s+", " ", match).strip().lower()
            found.setdefault(key, []).append(chunk)

    if not found:
        print("No fault codes found in the corpus. That is almost certainly wrong.")
        return 1

    print(f"{len(found)} distinct codes across {len(chunks)} chunks")
    print()
    orphans = 0
    for code in sorted(found):
        holders = found[code]
        # A code with only a handful of words beside it has lost its meaning.
        best = max(holders, key=lambda c: c.word_count)
        flag = "" if best.word_count >= 20 else "  ORPHAN"
        if flag:
            orphans += 1
        if args.verbose or flag:
            print(f"  {code:<14} in {len(holders):>3} chunks  best={best.word_count:>3}w{flag}")
    print()
    print(f"{orphans} codes appear only in chunks too small to carry a description")
    return 1 if orphans else 0


def cmd_suspect(chunks: list[Chunk], args: argparse.Namespace) -> int:
    """Surface chunks that look wrong without reading all of them.

    Aggregate statistics hide the failures that matter. These heuristics catch
    the shapes that mangled extraction produces.
    """
    del args
    tiny = [c for c in chunks if c.word_count < 20]
    huge = [c for c in chunks if c.word_count > 400]
    severed = [c for c in chunks if re.match(r"^[a-z]", c.text.split("\n", 1)[-1].strip())]
    # Very long lines of short tokens are the signature of spliced columns or a
    # table read as prose.
    salad = [
        c
        for c in chunks
        if c.word_count > 40
        and statistics.mean(len(w) for w in c.text.split()) < 3.4
    ]
    untitled = [c for c in chunks if c.section in {"UNTITLED", ""}]

    for label, group in (
        ("under 20 words", tiny),
        ("over 400 words", huge),
        ("starts mid-sentence", severed),
        ("possible column splice", salad),
        ("no section", untitled),
    ):
        print(f"{label:<24} {len(group)}")

    print()
    for label, group in (
        ("TINY", tiny),
        ("HUGE", huge),
        ("SEVERED", severed),
        ("SALAD", salad),
    ):
        for chunk in group[:2]:
            print(f"### {label}")
            show_chunk(chunk, full=False)

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect chunked manual content.")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("stats")

    p_show = sub.add_parser("show")
    p_show.add_argument("--id")
    p_show.add_argument("--page", type=int)
    p_show.add_argument("--section")
    p_show.add_argument("--limit", type=int, default=5)

    p_grep = sub.add_parser("grep")
    p_grep.add_argument("pattern")
    p_grep.add_argument("--limit", type=int, default=5)
    p_grep.add_argument("--brief", action="store_true")

    p_sample = sub.add_parser("sample")
    p_sample.add_argument("--n", type=int, default=5)
    p_sample.add_argument("--type")
    p_sample.add_argument("--seed", type=int, default=0)

    p_codes = sub.add_parser("codes")
    p_codes.add_argument("-v", "--verbose", action="store_true")

    sub.add_parser("suspect")

    args = parser.parse_args(argv)
    chunks = load(args.chunks)

    if args.command == "stats":
        return cmd_stats(chunks)
    if args.command == "show":
        return cmd_show(chunks, args)
    if args.command == "grep":
        return cmd_grep(chunks, args)
    if args.command == "sample":
        return cmd_sample(chunks, args)
    if args.command == "codes":
        return cmd_codes(chunks, args)
    if args.command == "suspect":
        return cmd_suspect(chunks, args)
    raise SystemExit(f"Unknown command {args.command}")


if __name__ == "__main__":
    sys.exit(main())
