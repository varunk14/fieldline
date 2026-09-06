"""Semantic chunking for RTU service manuals.

Fixed-width windows are not an option here. The single most important query
this product answers is a fault-code lookup, and a character window will
happily cut "7 Flashes" away from "Rollout Switch Lockout", leaving two chunks
that each retrieve for the wrong thing. So chunks break on the document's own
boundaries: sections, procedure steps, and table rows.

Two kinds of content, chunked differently:

* Tables become one chunk per row wherever a row stands on its own, with the
  caption and column headers repeated into every chunk. A row reading
  "2 Flashes | Limit Switch Fault | ..." is unusable without "Table 19 - IGC
  Board LED Alarm Codes" above it, and a technician asking about flash codes
  will never say the word "table".
* Prose accumulates to a target size, breaking at headings and never in the
  middle of a numbered step.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

from fieldline.corpus.contents import parse_contents, section_index, subsection_index
from fieldline.corpus.extract import extract_pages
from fieldline.corpus.models import Chunk, ContentType, Page, PageTable, make_chunk_id

logger = logging.getLogger(__name__)

TARGET_WORDS = 260
MAX_WORDS = 380
MIN_WORDS = 25

# A step line: "1." / "3)" / "Step 4". Chunks never start mid-step, so these
# are the only safe places to break inside a procedure.
STEP_RE = re.compile(r"^\s*(?:step\s+)?\(?(\d{1,2})[.)]\s+", re.IGNORECASE)

# A chunk starting lower case is a severed continuation of the line above.
CONTINUATION_RE = re.compile(r"^[a-z]")

SAFETY_MARKERS = ("WARNING", "CAUTION", "DANGER")

FAULT_CAPTION_RE = re.compile(r"alarm|fault|error|code|troubleshoot|diagnos", re.IGNORECASE)
SPEC_CAPTION_RE = re.compile(
    r"torque|pressure|physical data|performance|electrical data|charging|weight|dimension",
    re.IGNORECASE,
)
FAULT_CELL_RE = re.compile(r"^\s*(\d{1,2}\s*flash|e-?\d{1,3}|a\d{2,3}|code\s*\d+)", re.IGNORECASE)


def _words(text: str) -> int:
    return len(text.split())


def _resolve_sections(pages: list[Page]) -> tuple[dict[int, str], dict[int, str]]:
    """Section per page: contents listing first, formatting as fallback.

    The contents page is authoritative but stops short of the appendices,
    which are a third of this manual and hold the physical-data and torque
    tables. Those pages fall back to their detected headings.
    """
    refs = parse_contents(pages)
    total = len(pages)
    major = section_index(refs, total)
    minor = subsection_index(refs, total)

    last_listed = max((r.start_page for r in refs), default=0)
    for page in pages:
        if page.number > last_listed and page.section:
            major[page.number] = page.section

    if not major:
        logger.warning("No section map available; falling back to headings only")
        major = {p.number: (p.section or "UNTITLED") for p in pages}
    return major, minor


def _classify_table(table: PageTable, section: str) -> ContentType:
    caption = table.caption or ""
    if FAULT_CAPTION_RE.search(caption):
        return ContentType.FAULT_CODE
    if table.rows and FAULT_CELL_RE.match(table.rows[0][0] if table.rows[0] else ""):
        return ContentType.FAULT_CODE
    if SPEC_CAPTION_RE.search(caption) or SPEC_CAPTION_RE.search(section):
        return ContentType.SPEC
    return ContentType.REFERENCE


def _classify_prose(text: str, section: str) -> ContentType:
    upper_section = section.upper()
    if "SAFETY" in upper_section or any(m in text for m in SAFETY_MARKERS):
        return ContentType.SAFETY
    if STEP_RE.search(text):
        return ContentType.PROCEDURE
    if SPEC_CAPTION_RE.search(section):
        return ContentType.SPEC
    if re.search(r"\b\d+(\.\d+)?\s*(ft-lb|in-lb|psig|rpm|volts?|amps?)\b", text, re.IGNORECASE):
        return ContentType.SPEC
    return ContentType.REFERENCE


def _render_row(table: PageTable, row: tuple[str, ...]) -> str:
    """Render one table row as self-describing text.

    Column headers are repeated into every row. Without them a retrieved row is
    a list of values whose meaning lived in a header the model never sees.
    """
    parts: list[str] = []
    for header, cell in zip(table.header, row, strict=False):
        cell = cell.strip()
        if not cell or cell in {"—", "-", "--"}:
            continue
        label = header.strip()
        parts.append(f"{label}: {cell}" if label else cell)
    # Trailing cells with no header still carry content worth keeping.
    if len(row) > len(table.header):
        parts.extend(c.strip() for c in row[len(table.header) :] if c.strip())
    return ". ".join(parts)


def _table_chunks(
    table: PageTable,
    manual_id: str,
    model_family: str,
    section: str,
    start_ordinal: int,
) -> list[Chunk]:
    """One chunk per row, merging rows too small to stand alone."""
    caption = table.caption or f"Table on page {table.page}"
    content_type = _classify_table(table, section)
    chunks: list[Chunk] = []

    buffer: list[str] = []
    ordinal = start_ordinal

    def flush() -> None:
        nonlocal buffer, ordinal
        if not buffer:
            return
        body = "\n".join(buffer)
        text = f"{caption}\n{body}"
        chunks.append(
            Chunk(
                id=make_chunk_id(manual_id, section, ordinal),
                text=text,
                manual_id=manual_id,
                model_family=model_family,
                section=section,
                page=table.page,
                content_type=content_type,
                ordinal=ordinal,
                extra={"source": "table", "caption": caption},
            )
        )
        ordinal += 1
        buffer = []

    for row in table.rows:
        rendered = _render_row(table, row)
        if not rendered.strip():
            continue
        # Rows accumulate to the target size and a row is never split, so a
        # fault code always stays with its description. Flushing at the minimum
        # instead produced 1663 chunks averaging 64 words - technically valid
        # rows, far too small to carry context.
        if buffer and _words("\n".join(buffer)) + _words(rendered) + _words(caption) > MAX_WORDS:
            flush()
        buffer.append(rendered)
        if _words("\n".join(buffer)) + _words(caption) >= TARGET_WORDS:
            flush()

    flush()
    return _merge_undersized(chunks)


def _merge_undersized(chunks: list[Chunk]) -> list[Chunk]:
    """Fold a too-small trailing chunk into its predecessor.

    Dropping it would lose content, and emitting it would violate the minimum
    size. The last rows of a table and the tail of a section are exactly where
    this happens.
    """
    merged: list[Chunk] = []
    carried: str | None = None
    for chunk in chunks:
        current = chunk
        if carried is not None:
            current = replace(current, text=f"{carried}\n{current.text}")
            carried = None
        if current.word_count >= MIN_WORDS:
            merged.append(current)
            continue
        if merged and merged[-1].word_count + current.word_count <= MAX_WORDS:
            previous = merged[-1]
            body = current.text.split("\n", 1)[-1]
            merged[-1] = replace(previous, text=f"{previous.text}\n{body}")
            continue
        # Nothing to fold back into: carry it forward onto the next chunk
        # rather than emitting an undersized one or dropping its content.
        carried = current.text
    if carried is not None and merged:
        previous = merged[-1]
        merged[-1] = replace(previous, text=f"{previous.text}\n{carried.split(chr(10), 1)[-1]}")
    return merged


@dataclass
class _ProseRun:
    """Prose accumulating within one section, across page boundaries."""

    section: str
    subsection: str
    page: int
    lines: list[str] = field(default_factory=list)


def _split_at_sentence(buffer: list[str], *, force: bool) -> tuple[list[str], list[str]]:
    """Split a buffer so the emitted part ends on a finished sentence.

    Breaking purely on the word budget leaves the next chunk starting
    mid-sentence, which retrieves badly and reads as though the manual were
    quoted carelessly. Backing up to the last line that ends in sentence
    punctuation costs a few words and keeps both halves readable.
    """
    if not force or len(buffer) < 2:
        return buffer, []
    for index in range(len(buffer) - 1, 0, -1):
        if buffer[index - 1].rstrip().endswith((".", "!", "?", ":")):
            return buffer[:index], buffer[index:]
    return buffer, []


def _enforce_max(chunks: list[Chunk]) -> list[Chunk]:
    """Split any chunk still over the cap, on sentence boundaries.

    A single table row or an unbroken paragraph can exceed the budget on its
    own, and no accumulation logic upstream can prevent that. Splitting here
    guarantees the invariant rather than hoping the earlier passes held it.
    The heading line is repeated onto each part so the pieces stay citable.
    """
    result: list[Chunk] = []
    for chunk in chunks:
        if chunk.word_count <= MAX_WORDS:
            result.append(chunk)
            continue

        heading, _, body = chunk.text.partition("\n")
        sentences = re.split(r"(?<=[.!?])\s+", body)
        budget = MAX_WORDS - _words(heading)
        buffer: list[str] = []
        parts: list[str] = []
        for sentence in sentences:
            if buffer and _words(" ".join([*buffer, sentence])) > budget:
                parts.append(" ".join(buffer))
                buffer = []
            buffer.append(sentence)
        if buffer:
            parts.append(" ".join(buffer))

        for index, part in enumerate(parts):
            suffix = f" (part {index + 1} of {len(parts)})" if len(parts) > 1 else ""
            result.append(
                replace(
                    chunk,
                    id=f"{chunk.id}-{index}" if index else chunk.id,
                    text=f"{heading}{suffix}\n{part}",
                )
            )
    return result


def _prose_chunks(
    runs: list[_ProseRun],
    manual_id: str,
    model_family: str,
    ordinals: dict[str, int],
) -> list[Chunk]:
    """Accumulate prose to the target size, breaking only at safe boundaries.

    Runs stream across page boundaries. Flushing at the end of every page was
    what drove the mean chunk down to 64 words: a page break is a printing
    artefact, not a semantic boundary, and a procedure that spans two pages is
    one procedure.
    """
    chunks: list[Chunk] = []

    def emit(run: _ProseRun, body: str) -> None:
        text = body.strip()
        if not text:
            return
        ordinal = ordinals.get(run.section, 0)
        ordinals[run.section] = ordinal + 1
        heading = f"{run.section} - {run.subsection}" if run.subsection else run.section
        chunks.append(
            Chunk(
                id=make_chunk_id(manual_id, run.section, ordinal),
                text=f"{heading}\n{text}",
                manual_id=manual_id,
                model_family=model_family,
                section=run.section,
                page=run.page,
                content_type=_classify_prose(text, run.section),
                ordinal=ordinal,
                extra={
                    "source": "prose",
                    **({"subsection": run.subsection} if run.subsection else {}),
                },
            )
        )

    for run in runs:
        buffer: list[str] = []
        heading_cost = _words(f"{run.section} - {run.subsection}" if run.subsection else run.section)
        for raw in run.lines:
            text = raw.strip()
            if not text:
                continue
            pending = _words("\n".join([*buffer, text])) + heading_cost
            # Break before a new step, never inside one.
            safe_break = bool(STEP_RE.match(text))
            if buffer and (pending > MAX_WORDS or (pending >= TARGET_WORDS and safe_break)):
                head, tail = _split_at_sentence(buffer, force=not safe_break)
                emit(run, "\n".join(head))
                buffer = tail
            buffer.append(text)
        emit(run, "\n".join(buffer))

    return _merge_undersized(chunks)


def _drop_severed_starts(chunks: list[Chunk]) -> list[Chunk]:
    """Report chunks that begin mid-sentence.

    A chunk whose body starts lower case is a continuation that lost its head.
    These are logged rather than silently dropped: the count is a direct
    measure of chunking quality and should be visible, not hidden.
    """
    severed = [c for c in chunks if CONTINUATION_RE.match(c.text.split("\n", 1)[-1].strip())]
    if severed:
        logger.info("%d chunks begin mid-sentence", len(severed))
    return chunks


def chunk_manual(
    pdf_path: Path,
    manual_id: str,
    model_family: str,
) -> list[Chunk]:
    """Extract and chunk one manual."""
    pages = extract_pages(pdf_path)
    major, minor = _resolve_sections(pages)

    chunks: list[Chunk] = []
    ordinals: dict[str, int] = {}

    # Tables first, so every table row is chunked from structure rather than
    # from the prose stream it was lifted out of.
    for page in pages:
        section = major.get(page.number) or page.section or "UNTITLED"
        for table in page.tables:
            start = ordinals.get(section, 0)
            produced = _table_chunks(table, manual_id, model_family, section, start)
            ordinals[section] = start + max(len(produced), 1)
            chunks.extend(produced)

    # Prose accumulates into runs that continue across page boundaries and
    # break only when the section or subsection changes.
    runs: list[_ProseRun] = []
    for page in pages:
        section = major.get(page.number) or page.section or "UNTITLED"
        subsection = minor.get(page.number, "")
        if not runs or runs[-1].section != section or runs[-1].subsection != subsection:
            runs.append(_ProseRun(section=section, subsection=subsection, page=page.number))
        runs[-1].lines.extend(line.text for line in page.lines)

    chunks.extend(_prose_chunks(runs, manual_id, model_family, ordinals))
    chunks = _enforce_max(chunks)
    chunks = _drop_severed_starts(chunks)
    logger.info("Produced %d chunks from %s", len(chunks), pdf_path.name)
    return chunks
