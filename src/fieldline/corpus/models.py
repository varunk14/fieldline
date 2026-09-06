"""Data model for extracted pages and the chunks derived from them."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ContentType(str, Enum):
    """What kind of question a chunk can answer.

    Used as a Moss metadata filter and to drive chunking policy: a fault code
    row and a multi-page procedure need different treatment.
    """

    FAULT_CODE = "fault_code"
    PROCEDURE = "procedure"
    SPEC = "spec"
    SAFETY = "safety"
    REFERENCE = "reference"


@dataclass(frozen=True)
class PageTable:
    """A table lifted off a page, with its caption and header row preserved.

    The caption matters as much as the cells. A row reading
    "2 Flashes | Limit Switch Fault | ..." is meaningless without
    "Table 19 - IGC Board LED Alarm Codes" above it, and a technician asking
    about flash codes will not use the word "table".
    """

    caption: str
    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    page: int


@dataclass(frozen=True)
class TextLine:
    """One visual line of prose, tagged with its heading level.

    Heading-ness is decided during extraction from font weight and size, where
    that information still exists. Recovering it later from the text alone is
    not possible: figure callouts inside diagrams are also set in capitals, and
    that is exactly how "ROOFTOP" and "BELT" first got mistaken for sections.

    Level 0 is body text. Levels 1 and 2 share a font and are told apart by
    case: the manual sets major sections in capitals ("TROUBLESHOOTING THE
    COOLING SYSTEM") and subsections in title case ("Replacing the Motor").
    """

    text: str
    heading_level: int = 0

    @property
    def is_heading(self) -> bool:
        return self.heading_level > 0


@dataclass(frozen=True)
class Page:
    """One extracted page.

    `lines` holds narrative prose in reading order with table regions removed;
    tables are carried separately in `tables` because they chunk differently.
    """

    number: int
    lines: tuple[TextLine, ...] = ()
    tables: tuple[PageTable, ...] = ()
    section: str | None = None

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)


@dataclass(frozen=True)
class Chunk:
    """One indexable unit of manual content."""

    id: str
    text: str
    manual_id: str
    model_family: str
    section: str
    page: int
    content_type: ContentType
    ordinal: int = 0
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def to_metadata(self) -> dict[str, str]:
        """Flatten to the string-valued metadata dict Moss stores."""
        meta = {
            "manual_id": self.manual_id,
            "model_family": self.model_family,
            "section": self.section,
            "page": str(self.page),
            "content_type": self.content_type.value,
        }
        meta.update(self.extra)
        return meta

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "manual_id": self.manual_id,
            "model_family": self.model_family,
            "section": self.section,
            "page": self.page,
            "content_type": self.content_type.value,
            "ordinal": self.ordinal,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chunk:
        return cls(
            id=str(data["id"]),
            text=str(data["text"]),
            manual_id=str(data["manual_id"]),
            model_family=str(data["model_family"]),
            section=str(data["section"]),
            page=int(data["page"]),
            content_type=ContentType(data["content_type"]),
            ordinal=int(data.get("ordinal", 0)),
            extra=dict(data.get("extra", {})),
        )


def make_chunk_id(manual_id: str, section: str, ordinal: int) -> str:
    """Derive a stable chunk id.

    Idempotency is a property of the id function, not of a convention someone
    has to remember: the same input always produces the same id, so
    re-ingestion cannot create duplicates even if the writer is careless.
    Page number is deliberately excluded - a chunk that shifts across a page
    boundary after a chunking tweak is still the same chunk.
    """
    digest = hashlib.sha256(f"{manual_id}|{section}|{ordinal}".encode()).hexdigest()
    return f"{manual_id}-{digest[:12]}"
