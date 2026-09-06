"""Runtime configuration, read from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHUNKS = REPO_ROOT / "corpus" / "chunks.json"


@dataclass(frozen=True)
class Settings:
    """Credentials and index naming."""

    moss_project_id: str
    moss_project_key: str
    moss_index_name: str

    @classmethod
    def from_env(cls, *, dotenv_path: Path | None = None) -> Settings:
        load_dotenv(dotenv_path or REPO_ROOT / ".env", override=False)
        missing = [
            name
            for name in ("MOSS_PROJECT_ID", "MOSS_PROJECT_KEY", "MOSS_INDEX_NAME")
            if not os.environ.get(name)
        ]
        if missing:
            raise SystemExit(
                f"Missing environment variables: {', '.join(missing)}. "
                "Copy .env.example to .env and fill them in."
            )
        return cls(
            moss_project_id=os.environ["MOSS_PROJECT_ID"],
            moss_project_key=os.environ["MOSS_PROJECT_KEY"],
            moss_index_name=os.environ["MOSS_INDEX_NAME"],
        )
