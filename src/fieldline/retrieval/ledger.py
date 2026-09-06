"""Append-only ledger of Moss ingest calls.

The Developer tier is a small monthly credit and it is not documented whether
that credit is consumed by ingest or by queries. Rather than guess, every call
that could spend it is recorded with a timestamp, so what the tier costs is
observable after the fact instead of inferred from a balance that has already
run out.

The ledger is local and never committed - it is operational data, not source.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_LEDGER = Path(os.environ.get("FIELDLINE_LEDGER", "corpus/moss_ledger.jsonl"))


@dataclass(frozen=True)
class IngestRecord:
    """One billable-looking Moss operation."""

    timestamp: str
    operation: str
    index_name: str
    doc_count: int
    duration_ms: float
    job_id: str = ""
    outcome: str = "ok"
    detail: str = ""


def record(
    operation: str,
    index_name: str,
    doc_count: int,
    duration_ms: float,
    *,
    job_id: str = "",
    outcome: str = "ok",
    detail: str = "",
    path: Path | None = None,
) -> IngestRecord:
    """Append one entry to the ledger and return it."""
    entry = IngestRecord(
        timestamp=datetime.now(UTC).isoformat(timespec="seconds"),
        operation=operation,
        index_name=index_name,
        doc_count=doc_count,
        duration_ms=round(duration_ms, 1),
        job_id=job_id,
        outcome=outcome,
        detail=detail,
    )
    target = path or DEFAULT_LEDGER
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(entry)) + "\n")
    logger.info("ledger: %s %s (%d docs, %.0fms)", operation, index_name, doc_count, duration_ms)
    return entry


def read(path: Path | None = None) -> list[IngestRecord]:
    """Read the ledger back, oldest first."""
    target = path or DEFAULT_LEDGER
    if not target.exists():
        return []
    entries: list[IngestRecord] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(IngestRecord(**json.loads(line)))
    return entries
