"""PDF extraction for two-column RTU service manuals.

Carrier's service manuals are laid out in two columns with full-width tables
and figures interrupting the flow. Reading such a page with a naive
left-to-right pass splices the columns together line by line, producing text
like "STAGED AIR VOLUME (SAV) CONTROL: 2- ADDITIONAL VFD INSTALLATION AND".
That reads as plausible prose to an aggregate word count and as nonsense to a
retriever, which is the failure this module exists to prevent.

The approach:

1. Pull tables out first and record their bounding boxes.
2. Split what remains into horizontal bands wherever a full-width element
   (a table, or a line of text spanning the gutter) crosses the page.
3. Within each band, detect the gutter and read the left column fully before
   the right one.
"""

from __future__ import annotations

import logging
import re
import statistics
from collections import Counter
from itertools import pairwise
from pathlib import Path
from typing import Any

import pdfplumber

from fieldline.corpus.models import Page, PageTable, TextLine

logger = logging.getLogger(__name__)

# A gutter must be at least this fraction of page width to count as a column
# separator rather than ordinary word spacing.
MIN_GUTTER_FRACTION = 0.02

# The gutter is looked for in the middle of the page only. A gap in the outer
# thirds is a margin or an indent, not a column boundary.
GUTTER_SEARCH_RANGE = (0.35, 0.65)

# Each column must hold at least this share of a band's words. Otherwise the
# "gutter" is just a ragged right edge above a short paragraph.
MIN_COLUMN_SHARE = 0.15

# Bands shorter than this are noise from figure labels and page furniture.
MIN_BAND_HEIGHT = 8.0

# Above this share of dot-leader words, a band is contents-page layout.
LEADER_DENSITY_THRESHOLD = 0.15

CAPTION_RE = re.compile(r"^\s*(Table|Fig\.?|Figure)\s+([A-Z0-9]+)\s*[-—–]\s*(.+)$", re.IGNORECASE)

# Section headings are set in Helvetica-Bold at 11pt. Body text is
# TimesNewRoman 10pt, table headers are bold 8pt, and figure callouts inside
# diagrams are ArialMT 7.5-8pt. Capitalisation alone does not separate them -
# callouts are capitalised too - so weight and size decide instead.
HEADING_MIN_SIZE = 10.5
BOLD_MARKERS = ("bold", "black", "heavy")

# How far past its column's left edge a heading may start. Body paragraphs and
# headings are both flush left in this manual; callouts are not.
HEADING_INDENT_TOLERANCE = 4.0

# Body text is 10pt. The labels printed inside figures are 7.5-8pt, and they sit
# at the same vertical positions as the prose beside them, so band splitting
# cannot separate them - they interleave into the text as stray words like
# "BELT" and "DEFLECTION" mid-sentence. Size excludes them cleanly.
PROSE_MIN_SIZE = 9.5

# A row must clear the gutter by this much on both sides to count as
# full-width rather than as a long line in one column.
FULL_WIDTH_MARGIN = 12.0

# The gutter's text coverage must be below this share of the page's typical
# coverage. Set loosely enough to survive the occasional word that overhangs.
GUTTER_COVERAGE_RATIO = 0.25

# A ruled table narrower than this share of the page probably has unruled
# columns outside its detected border.
NARROW_TABLE_RATIO = 0.75

# Lines that carry heading formatting but are safety-box or furniture labels.
HEADING_STOPWORDS = frozenset(
    {"LEGEND", "NOTES", "NOTE", "CONTENTS", "WARNING", "CAUTION", "DANGER", "IMPORTANT"}
)

WORD_ATTRS = ["fontname", "size"]


def _is_bold(word: dict[str, Any]) -> bool:
    return any(marker in str(word.get("fontname", "")).lower() for marker in BOLD_MARKERS)


def _is_heading_word(word: dict[str, Any]) -> bool:
    return _is_bold(word) and float(word.get("size", 0)) >= HEADING_MIN_SIZE


def _words_in_bbox(words: list[dict[str, Any]], top: float, bottom: float) -> list[dict[str, Any]]:
    return [w for w in words if w["top"] >= top - 0.5 and w["bottom"] <= bottom + 0.5]


def _column_margin(words: list[dict[str, Any]]) -> float | None:
    """The dominant left margin of a column.

    Deliberately the mode rather than the minimum. Page numbers and running
    furniture sit further left than the text margin, so a minimum puts the
    edge outside the body text and every real heading then fails the
    flush-left test - which collapsed section detection to almost nothing the
    first time this was tried.
    """
    if not words:
        return None
    starts = Counter(round(w["x0"]) for w in words)
    return float(starts.most_common(1)[0][0])


def _detect_gutter(words: list[dict[str, Any]], x0: float, x1: float) -> float | None:
    """Find the x coordinate of the column separator, if there is one.

    Works by projecting every word onto the x axis and looking for the widest
    uncovered band in the middle of the page.
    """
    if len(words) < 12:
        return None

    width = x1 - x0
    lo = x0 + width * GUTTER_SEARCH_RANGE[0]
    hi = x0 + width * GUTTER_SEARCH_RANGE[1]

    spans = sorted((w["x0"], w["x1"]) for w in words)
    merged: list[list[float]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    best: tuple[float, float] | None = None
    for left, right in pairwise(merged):
        gap_start, gap_end = left[1], right[0]
        gap_width = gap_end - gap_start
        midpoint = (gap_start + gap_end) / 2
        if not (lo <= midpoint <= hi):
            continue
        if gap_width < width * MIN_GUTTER_FRACTION:
            continue
        if best is None or gap_width > best[1]:
            best = (midpoint, gap_width)

    if best is None:
        return None

    gutter = best[0]
    left_count = sum(1 for w in words if w["x1"] <= gutter)
    right_count = sum(1 for w in words if w["x0"] >= gutter)
    total = len(words)
    if min(left_count, right_count) < total * MIN_COLUMN_SHARE:
        return None
    return gutter


def _lines_from_words(
    words: list[dict[str, Any]],
    column_x0: float | None = None,
    tolerance: float = 2.5,
) -> list[TextLine]:
    """Group words into visual lines, tagging each as heading or body.

    `column_x0` is the left edge of the column these words belong to. Section
    headings are flush with it; text set in the heading font that floats
    somewhere else on the page is a diagram callout, not a heading. Font alone
    could not tell those apart - it kept promoting labels like "9 CELL" out of
    the middle of a burner illustration.
    """
    if not words:
        return []
    ordered = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))
    groups: list[list[dict[str, Any]]] = [[ordered[0]]]
    for word in ordered[1:]:
        if abs(word["top"] - groups[-1][-1]["top"]) <= tolerance:
            groups[-1].append(word)
        else:
            groups.append([word])

    lines: list[TextLine] = []
    for group in groups:
        ordered_group = sorted(group, key=lambda w: w["x0"])
        text = " ".join(w["text"] for w in ordered_group)
        # A line counts as a heading when most of its words carry heading
        # formatting, which tolerates a stray inline symbol.
        heading_words = sum(1 for w in ordered_group if _is_heading_word(w))
        flush_left = (
            column_x0 is None
            or abs(ordered_group[0]["x0"] - column_x0) <= HEADING_INDENT_TOLERANCE
        )
        if heading_words * 2 > len(ordered_group) and flush_left:
            level = 1 if text.upper() == text else 2
        else:
            level = 0
        lines.append(TextLine(text=text, heading_level=level))
    return lines


def _group_rows(words: list[dict[str, Any]], tolerance: float = 2.5) -> list[list[dict[str, Any]]]:
    """Group words into visual rows across the whole page width."""
    if not words:
        return []
    ordered = sorted(words, key=lambda w: (round(w["top"], 1), w["x0"]))
    rows: list[list[dict[str, Any]]] = [[ordered[0]]]
    for word in ordered[1:]:
        if abs(word["top"] - rows[-1][-1]["top"]) <= tolerance:
            rows[-1].append(word)
        else:
            rows.append([word])
    return rows


def _estimate_gutter(words: list[dict[str, Any]], x0: float, x1: float) -> float | None:
    """Estimate the column separator by minimum text coverage.

    `_detect_gutter` requires a band no word crosses, which a page with a
    full-width notice box does not have. This looks instead for the x with the
    least coverage in the middle of the page, so the gutter is still found and
    the full-width elements can then be split out as their own bands.
    """
    if len(words) < 20:
        return None

    width = x1 - x0
    lo = x0 + width * GUTTER_SEARCH_RANGE[0]
    hi = x0 + width * GUTTER_SEARCH_RANGE[1]

    coverage: Counter[int] = Counter()
    for word in words:
        for bucket in range(int(word["x0"]), int(word["x1"]) + 1):
            coverage[bucket] += 1

    candidates = [(coverage.get(b, 0), b) for b in range(int(lo), int(hi) + 1)]
    if not candidates:
        return None
    best_count, best_x = min(candidates)

    typical = statistics.median(
        [coverage.get(b, 0) for b in range(int(x0), int(x1) + 1)] or [0]
    )
    if typical == 0 or best_count > typical * GUTTER_COVERAGE_RATIO:
        return None

    gutter = float(best_x)
    left = sum(1 for w in words if w["x1"] <= gutter)
    right = sum(1 for w in words if w["x0"] >= gutter)
    if min(left, right) < len(words) * MIN_COLUMN_SHARE:
        return None
    return gutter


def _leader_density(words: list[dict[str, Any]]) -> float:
    """Share of words that are dot-leader runs, as on a contents page."""
    if not words:
        return 0.0
    leaders = sum(1 for w in words if set(w["text"]) <= {".", "…", " "} and w["text"].strip())
    return leaders / len(words)


def _read_band(
    words: list[dict[str, Any]],
    x0: float,
    x1: float,
    forced_gutter: float | None = None,
) -> list[TextLine]:
    """Read one horizontal band, honouring columns if present."""
    gutter = forced_gutter if forced_gutter is not None else _detect_gutter(words, x0, x1)

    if gutter is None:
        return _lines_from_words(words, _column_margin(words))

    left = [w for w in words if w["x1"] <= gutter]
    right = [w for w in words if w["x0"] >= gutter]
    # Words straddling the gutter belong to neither column; keep them with the
    # side holding most of their width so nothing is silently dropped.
    for word in words:
        if word["x1"] > gutter > word["x0"]:
            if gutter - word["x0"] >= word["x1"] - gutter:
                left.append(word)
            else:
                right.append(word)

    left_x0 = _column_margin(left)
    right_x0 = _column_margin(right)
    return _lines_from_words(left, left_x0) + _lines_from_words(right, right_x0)


def _clean_cell(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _find_caption(table_top: float, words: list[dict[str, Any]]) -> str:
    """Caption is the nearest 'Table N - Title' line above the table."""
    above = [w for w in words if w["bottom"] <= table_top + 2]
    for line in reversed(_lines_from_words(above)):
        if CAPTION_RE.match(line.text):
            return re.sub(r"\s+", " ", line.text).strip()
    return ""


def _widen_table(
    page: pdfplumber.page.Page,
    rows: list[tuple[str, ...]],
    bbox: tuple[float, float, float, float],
    found_rows: list[Any],
) -> list[tuple[str, ...]]:
    """Recover columns that ruling lines do not enclose.

    Carrier rules only some of a table's columns. On the IGC alarm code table
    the ruled region covers DESCRIPTION, ACTION and RESET METHOD but not the
    LED FLASH CODE column on the left or PROBABLE CAUSE on the right, so
    line-based detection silently returns a table missing the fault code
    itself - every row reading "Limit Switch Fault" with nothing to look it up
    by. Re-reading the same rows across the full page width with text-based
    column detection recovers them.
    """
    left, top, right, bottom = bbox
    page_left, page_right = float(page.bbox[0]), float(page.bbox[2])
    page_width = page_right - page_left
    if page_width <= 0 or (right - left) / page_width >= NARROW_TABLE_RATIO:
        return rows

    words = [
        w
        for w in page.extract_words()
        if top - 1 <= w["top"] and w["bottom"] <= bottom + 1
    ]
    if not words:
        return rows

    def side_text(row_top: float, row_bottom: float, *, before: bool) -> str:
        picked = [
            w
            for w in words
            if row_top - 1 <= w["top"]
            and w["bottom"] <= row_bottom + 1
            and (w["x1"] <= left + 1 if before else w["x0"] >= right - 1)
        ]
        if not picked:
            return ""
        ordered = sorted(picked, key=lambda w: (round(w["top"], 1), w["x0"]))
        return re.sub(r"\s+", " ", " ".join(w["text"] for w in ordered)).strip()

    try:
        row_boxes = [r.bbox for r in found_rows]
    except AttributeError:  # pragma: no cover - pdfplumber shape change
        return rows

    if len(row_boxes) != len(rows):
        return rows

    widened: list[tuple[str, ...]] = []
    gained_left = gained_right = False
    for row, box in zip(rows, row_boxes, strict=True):
        _, row_top, _, row_bottom = box
        prefix = side_text(row_top, row_bottom, before=True)
        suffix = side_text(row_top, row_bottom, before=False)
        gained_left = gained_left or bool(prefix)
        gained_right = gained_right or bool(suffix)
        widened.append(
            tuple(
                cell
                for cell in ([prefix] if prefix else []) + list(row) + ([suffix] if suffix else [])
            )
        )

    if not (gained_left or gained_right):
        return rows

    logger.info(
        "Recovered unruled columns on page %s (left=%s right=%s)",
        page.page_number,
        gained_left,
        gained_right,
    )
    return widened


def _extract_tables(page: pdfplumber.page.Page) -> tuple[list[PageTable], list[tuple[float, float]]]:
    """Return structured tables plus the vertical bands they occupy."""
    tables: list[PageTable] = []
    bands: list[tuple[float, float]] = []
    words = page.extract_words(extra_attrs=WORD_ATTRS)

    for found in page.find_tables():
        try:
            raw = found.extract()
        except (ValueError, TypeError, IndexError, AttributeError):
            # pdfplumber raises a variety of these on malformed table geometry.
            # A page whose table will not parse still has usable prose, so this
            # skips the table rather than the page.
            logger.warning("Table extraction failed on page %s", page.page_number)
            continue
        rows = [tuple(_clean_cell(c) for c in row) for row in raw if row]
        rows = [r for r in rows if any(cell for cell in r)]
        if len(rows) < 2:
            continue

        left, top, right, bottom = found.bbox
        rows = _widen_table(page, rows, (left, top, right, bottom), list(found.rows))
        caption = _find_caption(top, words)
        tables.append(
            PageTable(
                caption=caption,
                header=rows[0],
                rows=tuple(rows[1:]),
                page=page.page_number,
            )
        )
        bands.append((top, bottom))
    return tables, bands


def _narrative_lines(
    page: pdfplumber.page.Page, table_bands: list[tuple[float, float]]
) -> list[TextLine]:
    """Read the page's prose, skipping table regions and honouring columns."""
    words = page.extract_words(extra_attrs=WORD_ATTRS)
    if not words:
        return []

    def in_table(word: dict[str, Any]) -> bool:
        centre = (word["top"] + word["bottom"]) / 2
        return any(top <= centre <= bottom for top, bottom in table_bands)

    prose = [
        w
        for w in words
        if not in_table(w) and float(w.get("size", 0)) >= PROSE_MIN_SIZE
    ]
    if not prose:
        return []

    x0, x1 = float(page.bbox[0]), float(page.bbox[2])

    # A contents page is two columns, but its dot leaders run right across the
    # gutter so there is no empty band to detect. The layout is a fixed grid,
    # so once the page is recognised by its leaders, split every band down the
    # middle. Without this the columns interleave line by line and entries fuse
    # into inventions like "ADDITIONAL VFD INSTALLATION AND PROTECTIVE DEVICES".
    forced_gutter = (
        (x0 + x1) / 2 if _leader_density(prose) >= LEADER_DENSITY_THRESHOLD else None
    )

    gutter = forced_gutter if forced_gutter is not None else _estimate_gutter(prose, x0, x1)
    if gutter is None:
        return [ln for ln in _lines_from_words(prose, _column_margin(prose)) if ln.text.strip()]

    # Bands are separated by full-width elements, not only by tables. Safety
    # boxes and notices span both columns, and because they cover the gutter
    # the whole page then looks single-column - which is how a two-column page
    # came out as "3. Reset the sensor ... 1. Press the controller's
    # test/reset switch", two columns spliced line by line.
    rows = _group_rows(prose)
    lines: list[TextLine] = []
    band: list[dict[str, Any]] = []

    for row in rows:
        row_x0 = min(w["x0"] for w in row)
        row_x1 = max(w["x1"] for w in row)
        spans_gutter = row_x0 < gutter - FULL_WIDTH_MARGIN and row_x1 > gutter + FULL_WIDTH_MARGIN
        if spans_gutter:
            if band:
                lines.extend(_read_band(band, x0, x1, gutter))
                band = []
            lines.extend(_lines_from_words(row, _column_margin(row)))
        else:
            band.extend(row)

    if band:
        lines.extend(_read_band(band, x0, x1, gutter))

    return [line for line in lines if line.text.strip()]


def _is_furniture(text: str) -> bool:
    """Heading-formatted text that is not a section title.

    Safety boxes ("WARNING"), figure annotations ("NOTE: SPARK GAP MUST BE AT
    THE BOTTOM OF THE BURNER") and diagram callout letters ("B D C F E A") are
    all set in the heading font. They must be demoted before headings are
    merged, or they fuse onto the real title next to them and produce sections
    like "CONVENIENCE OUTLETS WARNING".
    """
    stripped = text.strip().rstrip(":.")
    if not stripped:
        return True
    if stripped.upper() in HEADING_STOPWORDS:
        return True
    if re.match(r"^(NOTE|NOTES|IMPORTANT|LEGEND)\b\s*:", stripped, re.IGNORECASE):
        return True
    words = stripped.split()
    # Callout letter runs: mostly single characters, no real words.
    return bool(words) and sum(1 for w in words if len(w.strip(".,")) <= 1) * 2 > len(words)


def _is_barrier(text: str) -> bool:
    """A furniture label that must not merge with an adjacent real heading.

    Ordering matters here and the two cases pull opposite ways. "CONVENIENCE
    OUTLETS" followed by a "WARNING" box label must not merge, so the label has
    to be recognised before merging. But "NOTE: SPARK GAP MUST BE AT" wraps
    onto "THE BOTTOM OF THE BURNER", and demoting the first line before the
    merge strands the second as a section of its own. So barriers block
    merging, and everything else is demoted after the merge has run.
    """
    return text.strip().rstrip(":.").upper() in HEADING_STOPWORDS


def _demote_furniture(lines: list[TextLine]) -> list[TextLine]:
    return [
        TextLine(text=line.text, heading_level=0)
        if line.is_heading and _is_furniture(line.text)
        else line
        for line in lines
    ]


def _merge_heading_runs(lines: list[TextLine]) -> list[TextLine]:
    """Join consecutive heading lines into one heading.

    Section titles wrap across up to three lines ("STAGED AIR VOLUME (SAV)
    CONTROL: 2-" / "SPEED FAN WITH VARIABLE FREQUENCY" / "DRIVE (VFD)"), and a
    third of a title is not a usable citation.

    Only lines at the same level are joined. A major heading immediately
    followed by its subsection is two headings, not one: fusing them produced
    "COMPRESSORS Lubrication", which is neither.
    """
    merged: list[TextLine] = []
    for line in lines:
        if (
            line.is_heading
            and merged
            and merged[-1].heading_level == line.heading_level
            and not _is_barrier(line.text)
            and not _is_barrier(merged[-1].text)
        ):
            joined = f"{merged[-1].text} {line.text}"
            merged[-1] = TextLine(
                text=re.sub(r"\s+", " ", joined).strip(),
                heading_level=line.heading_level,
            )
        else:
            merged.append(line)
    return merged


def _is_section_heading(line: TextLine) -> bool:
    if not line.is_heading:
        return False
    text = line.text.strip().rstrip(":.")
    if not text or len(text) > 90:
        return False
    if text.upper() in HEADING_STOPWORDS:
        return False
    return sum(c.isalpha() for c in text) >= 4


def extract_pages(pdf_path: Path) -> list[Page]:
    """Extract every page of a manual, column-aware and table-aware."""
    pages: list[Page] = []
    current_section: str | None = None

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables, bands = _extract_tables(page)
            lines = _demote_furniture(_merge_heading_runs(_narrative_lines(page, bands)))

            # The section a page *starts* in is what its first chunks belong to;
            # headings further down open new sections, which chunking tracks.
            page_section = current_section
            for line in lines:
                if line.heading_level == 1 and _is_section_heading(line):
                    current_section = line.text.strip().rstrip(":.")
                    if page_section is None:
                        page_section = current_section

            pages.append(
                Page(
                    number=page.page_number,
                    lines=tuple(lines),
                    tables=tuple(tables),
                    section=page_section,
                )
            )

    logger.info("Extracted %d pages from %s", len(pages), pdf_path.name)
    return pages
