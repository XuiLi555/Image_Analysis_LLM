"""Runtime configuration.

Settings are read from environment variables (optionally loaded from a local
``.env`` file if ``python-dotenv`` is installed). Nothing here is secret by
default — API keys live only in your environment, never in the repo.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Load a .env file if python-dotenv is available (optional convenience).
try:  # pragma: no cover - trivial
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


# NOTE: model names change over time. Check the current list at
# https://docs.claude.com/en/docs/about-claude/models  and override with the
# IMAGELENS_MODEL environment variable if needed.
DEFAULT_MODEL = "claude-sonnet-4-5"


@dataclass
class Settings:
    """Resolved configuration for a run."""

    provider: str = "anthropic"
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    max_tokens: int = 1024
    # Longest edge (px) images are resized to before upload. Keeps requests
    # small and fast; vision models rarely benefit from more than ~1568px.
    max_image_edge: int = 1568

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            provider=os.getenv("IMAGELENS_PROVIDER", "anthropic"),
            model=os.getenv("IMAGELENS_MODEL", DEFAULT_MODEL),
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_tokens=int(os.getenv("IMAGELENS_MAX_TOKENS", "1024")),
            max_image_edge=int(os.getenv("IMAGELENS_MAX_IMAGE_EDGE", "1568")),
        )
