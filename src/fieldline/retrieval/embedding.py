"""Local embedding.

We generate embeddings ourselves rather than using Moss's bundled foundation
models. That is not a preference: as of 2026-09-06 any index built with a Moss
foundation model cannot be queried at all - the local runtime rejects the
model artifact and the cloud query endpoint returns 503 - while indexes built
with caller-supplied vectors work normally at ~5ms.

The consequence for how this build is described matters and is easy to get
wrong. Moss here is a fast local vector store, not an embedding-inclusive
retrieval engine. The sub-10ms local query claim still holds. The "no separate
embedding step" framing does not, and must not appear in the README, the
benchmark or the demo.

Model choice is fixed through MVP 6. The index cannot be cheaply rebuilt - the
Developer tier caps a project at three indexes - so changing embedding model
means rebuilding, and that is not a casual decision.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from fastembed import TextEmbedding

logger = logging.getLogger(__name__)

# Locked for the project. Chosen over all-MiniLM-L6-v2 because retrieval recall
# is the binding constraint (the MVP 2 gate is recall@3 >= 0.85 and the agent
# only ever sees three chunks), and because bge is trained for asymmetric
# retrieval - a short spoken question against a long manual passage - which is
# exactly this product's query shape. Costs about 2.4ms more per query than
# MiniLM, against an 800ms turn budget.
MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

# bge models expect this prefix on the query side only. Documents are embedded
# bare. Omitting it measurably degrades retrieval, and it is silent when wrong.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _model() -> TextEmbedding:
    """Load the model once per process.

    Cold load is several seconds, so it must happen at boot rather than on the
    first voice turn.
    """
    logger.info("Loading embedding model %s", MODEL_NAME)
    return TextEmbedding(model_name=MODEL_NAME)


def warm_up() -> None:
    """Force model load and one inference, so the first real query is fast."""
    embed_query("warm up")


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed corpus chunks. No query prefix - these are the passages."""
    if not texts:
        return []
    return [vector.tolist() for vector in _model().embed(texts)]


def embed_query(text: str) -> list[float]:
    """Embed one query, with the asymmetric-retrieval prefix applied."""
    vector = next(iter(_model().embed([QUERY_PREFIX + text])))
    return list(vector.tolist())
