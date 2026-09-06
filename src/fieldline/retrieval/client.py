"""Moss retrieval, with the in-process guarantee enforced rather than assumed.

The whole latency argument rests on one property: after `load_index`, queries
never leave this process. The SDK does not enforce it. `query()` checks whether
the index is loaded locally and, if it is not, silently falls back to the cloud
HTTP API - no exception, no warning, just a hundredfold slower answer and any
metadata filter quietly discarded.

That is the worst available failure mode: a working agent whose central claim
is false. So `MossRetriever.load()` probes after loading and refuses to come up
if the probe is too slow to have been served locally.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from moss import DocumentInfo, MossClient, QueryOptions

from fieldline.corpus.models import Chunk
from fieldline.retrieval import embedding, ledger

logger = logging.getLogger(__name__)

# A local query is single-digit milliseconds; the cloud fallback is 100-500ms.
# Anything past this threshold did not come from memory.
LOCAL_QUERY_MAX_MS = 40.0

# Blend of semantic and keyword matching. Below 1.0 keeps lexical signal, which
# is what makes "7 flashes" and model numbers findable; those are strings where
# embeddings are weakest and exact matching is exactly right.
DEFAULT_ALPHA = 0.8

# The agent only ever sees three chunks, per the Groq token budget.
DEFAULT_TOP_K = 3

PROBE_QUERY = "rollout switch lockout"


class IndexNotLoadedError(RuntimeError):
    """Raised when retrieval would be served by the cloud instead of memory."""


@dataclass(frozen=True)
class RetrievedChunk:
    """One search hit."""

    id: str
    text: str
    score: float
    metadata: dict[str, str]

    @property
    def section(self) -> str:
        return self.metadata.get("section", "")

    @property
    def page(self) -> str:
        return self.metadata.get("page", "")

    @property
    def manual_id(self) -> str:
        return self.metadata.get("manual_id", "")


@dataclass(frozen=True)
class Retrieval:
    """The result of one query."""

    query: str
    docs: list[RetrievedChunk]
    time_taken_ms: float

    @property
    def top_score(self) -> float:
        return self.docs[0].score if self.docs else 0.0


def chunks_to_documents(chunks: list[Chunk]) -> list[DocumentInfo]:
    """Convert corpus chunks into Moss documents, embedding them locally.

    Embedded in one batch rather than per chunk: batching is several times
    faster and this runs over the whole corpus.
    """
    vectors = embedding.embed_documents([c.text for c in chunks])
    return [
        DocumentInfo(id=c.id, text=c.text, metadata=c.to_metadata(), embedding=v)
        for c, v in zip(chunks, vectors, strict=True)
    ]


class MossRetriever:
    """Thin wrapper over MossClient that guarantees in-process retrieval."""

    def __init__(self, project_id: str, project_key: str, index_name: str) -> None:
        if not project_id or not project_key:
            raise ValueError("Moss credentials missing. See .env.example.")
        self._client = MossClient(project_id, project_key)
        self._index_name = index_name
        self._loaded = False

    @property
    def index_name(self) -> str:
        return self._index_name

    def is_loaded(self) -> bool:
        return self._loaded

    async def load(self, *, probe: bool = True) -> None:
        """Load the index into this process and prove it landed there.

        `probe` exists only so tests can load without asserting latency on a
        machine under load. Production paths must leave it on.
        """
        started = time.perf_counter()
        await self._client.load_index(self._index_name)
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info("Loaded index %s in %.0fms", self._index_name, elapsed_ms)

        if probe:
            result = await self._raw_query(PROBE_QUERY, top_k=1, alpha=DEFAULT_ALPHA)
            if result.time_taken_ms > LOCAL_QUERY_MAX_MS:
                raise IndexNotLoadedError(
                    f"Probe query took {result.time_taken_ms:.0f}ms, over the "
                    f"{LOCAL_QUERY_MAX_MS:.0f}ms ceiling for an in-memory index. "
                    "Moss is serving this from the cloud API, so the latency claim "
                    "does not hold and metadata filters are being ignored."
                )
            logger.info("Probe query served in %.1fms", result.time_taken_ms)

        self._loaded = True

    async def aquery(
        self,
        text: str,
        top_k: int = DEFAULT_TOP_K,
        alpha: float = DEFAULT_ALPHA,
        content_type: str | None = None,
    ) -> Retrieval:
        """Query the loaded index."""
        if not self._loaded:
            raise IndexNotLoadedError("load() must succeed before querying.")
        return await self._raw_query(text, top_k=top_k, alpha=alpha, content_type=content_type)

    async def _raw_query(
        self,
        text: str,
        top_k: int,
        alpha: float,
        content_type: str | None = None,
    ) -> Retrieval:
        # The index is built from caller-supplied vectors, so every query must
        # supply one too. Without it Moss cannot embed the query itself and
        # fails rather than falling back.
        options: dict[str, Any] = {
            "top_k": top_k,
            "alpha": alpha,
            "embedding": embedding.embed_query(text),
        }
        if content_type:
            # Filtering only works on a locally loaded index; the cloud path
            # discards it with a warning. load() has already proven we are local.
            options["filter"] = {
                "$and": [{"field": "content_type", "condition": {"$eq": content_type}}]
            }

        result = await self._client.query(self._index_name, text, QueryOptions(**options))
        docs = [
            RetrievedChunk(
                id=doc.id,
                text=doc.text,
                score=float(doc.score),
                metadata=dict(doc.metadata or {}),
            )
            for doc in result.docs
        ]
        return Retrieval(
            query=text,
            docs=docs,
            time_taken_ms=float(getattr(result, "time_taken_ms", 0.0)),
        )

    async def delete_index(self) -> None:
        """Drop the index. Recorded, because it frees one of three tier slots."""
        started = time.perf_counter()
        await self._client.delete_index(self._index_name)
        self._loaded = False
        ledger.record(
            "delete_index",
            self._index_name,
            0,
            (time.perf_counter() - started) * 1000,
        )

    async def create_index(self, chunks: list[Chunk]) -> str:
        """Build the index from chunks. Records the call in the ledger.

        This is the operation whose credit cost is undocumented, so it is
        logged whether it succeeds or fails - a failed build may still have
        spent the credit, and that is exactly the case worth having a record of.
        """
        documents = chunks_to_documents(chunks)
        started = time.perf_counter()
        try:
            result = await self._client.create_index(self._index_name, documents)
        except Exception as error:
            ledger.record(
                "create_index",
                self._index_name,
                len(documents),
                (time.perf_counter() - started) * 1000,
                outcome="failed",
                detail=str(error)[:300],
            )
            raise

        elapsed_ms = (time.perf_counter() - started) * 1000
        job_id = str(getattr(result, "job_id", ""))
        ledger.record(
            "create_index",
            self._index_name,
            len(documents),
            elapsed_ms,
            job_id=job_id,
        )
        logger.info(
            "Built index %s with %d docs in %.0fms", self._index_name, len(documents), elapsed_ms
        )
        return job_id
