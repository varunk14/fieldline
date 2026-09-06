"""Section map derived from a manual's own table of contents.

Detecting sections from heading formatting alone does not survive contact with
these documents. Headings, table column headers and the callout labels printed
inside diagrams share fonts and capitalisation, and the left margin that would
separate them is unreliable: on one page the dominant left edge belongs to an
undetected table's cells, on another to a figure's label column.

The manual states its own section list on the contents page, with a page number
against each entry. Assigning every page a section by page range removes the
guesswork: sections become exactly what Carrier says they are, which is also
what a technician sees if they open the paper copy.

Falls back to formatting-based headings when a manual has no parseable
contents page, so this is an improvement rather than a new dependency.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from fieldline.corpus.models import Page

logger = logging.getLogger(__name__)

# "SAFETY CONSIDERATIONS . . . . . . . 2" and the title-case subsection form.
# Dot leaders may be spaced or solid, and several entries share one line when
# the two contents columns interleave.
ENTRY_RE = re.compile(r"([A-Za-z][^.]*?)\s*[.…]\s*(?:[.…]\s*){2,}(\d{1,3})\b")

# A contents page names many sections; a body page mentioning one figure does
# not. Requiring several entries stops a stray dot-leader line being mistaken
# for a contents page.
MIN_ENTRIES_PER_PAGE = 5

# How far into the document to look. Contents sit at the front.
MAX_CONTENTS_PAGES = 6


@dataclass(frozen=True)
class SectionRef:
    """One contents entry: where a section starts and how deeply it nests."""

    title: str
    level: int
    start_page: int


def _clean_title(raw: str) -> str:
    title = re.sub(r"\s+", " ", raw).strip(" .…-")
    # Interleaved columns leave the previous entry's tail glued to the front of
    # the next one. Anything before a run of dot leaders belongs to the entry
    # that already claimed its page number.
    return title.strip()


def _level_of(title: str) -> int:
    letters = [c for c in title if c.isalpha()]
    if not letters:
        return 2
    upper = sum(1 for c in letters if c.isupper())
    return 1 if upper == len(letters) else 2


def parse_contents(pages: list[Page]) -> list[SectionRef]:
    """Read the contents page(s) into an ordered section map."""
    refs: list[SectionRef] = []

    for page in pages[:MAX_CONTENTS_PAGES]:
        matches: list[SectionRef] = []
        # Long titles wrap, and only the last line carries the page number.
        # Lines with no entry on them are held and prepended to the next one:
        # "STAGED AIR VOLUME (SAV) CONTROL: 2-SPEED" / "FAN WITH VARIABLE
        # FREQUENCY" / "DRIVE (VFD) . . . 6" is one section, not one third of
        # one. This is only safe because the columns are separated first.
        pending: list[str] = []
        seen_entry = False
        for line in page.lines:
            found = ENTRY_RE.findall(line.text)
            if not found:
                stripped = line.text.strip()
                # Buffering only starts after the first real entry. Everything
                # above it is the cover block ("Service and Maintenance
                # Instructions", "CONTENTS", "Page"), which otherwise gets glued
                # onto the first section title and, by introducing lower case,
                # demotes it to a subsection.
                if seen_entry and stripped and not any(c.isdigit() for c in stripped):
                    pending.append(stripped)
                continue
            seen_entry = True

            for position, (raw_title, raw_page) in enumerate(found):
                title = _clean_title(raw_title)
                if position == 0 and pending:
                    title = _clean_title(" ".join([*pending, title]))
                if len(title) < 3 or not any(c.isalpha() for c in title):
                    continue
                try:
                    start = int(raw_page)
                except ValueError:  # pragma: no cover - regex guarantees digits
                    continue
                matches.append(
                    SectionRef(title=title, level=_level_of(title), start_page=start)
                )
            pending = []

        if len(matches) >= MIN_ENTRIES_PER_PAGE:
            refs.extend(matches)

    if not refs:
        logger.warning("No parseable contents page found; falling back to headings")
        return []

    # Contents columns interleave, so entries arrive out of order.
    refs.sort(key=lambda r: (r.start_page, r.level))
    logger.info("Parsed %d contents entries", len(refs))
    return refs


def _build_index(refs: list[SectionRef], level: int, total_pages: int) -> dict[int, str]:
    entries = [r for r in refs if r.level == level]
    if not entries:
        return {}

    # Where several sections start on one page, the last one wins: it is the
    # one the bulk of that page's text belongs to.
    by_start: dict[int, str] = {}
    for ref in entries:
        by_start[ref.start_page] = ref.title

    index: dict[int, str] = {}
    current = ""
    # Runs to total_pages, not to the last contents entry. Appendices sit past
    # the end of the contents listing and would otherwise have no section.
    for page_no in range(1, total_pages + 1):
        if page_no in by_start:
            current = by_start[page_no]
        if current:
            index[page_no] = current
    return index


def section_index(refs: list[SectionRef], total_pages: int) -> dict[int, str]:
    """Map every page number to the major section it falls in."""
    return _build_index(refs, level=1, total_pages=total_pages)


def subsection_index(refs: list[SectionRef], total_pages: int) -> dict[int, str]:
    """Map every page number to the most recent subsection heading."""
    return _build_index(refs, level=2, total_pages=total_pages)
