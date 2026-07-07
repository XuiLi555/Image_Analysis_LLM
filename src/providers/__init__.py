"""Vision-LLM backends.

Every provider implements :class:`~imagelens.providers.base.Provider`, so the
rest of the code never depends on a specific vendor. ``get_provider`` is a
small factory that resolves a provider by name.
"""

from .base import Provider
from .anthropic_provider import AnthropicProvider
from .mock_provider import MockProvider


def get_provider(name: str, settings) -> Provider:
    """Return an instantiated provider for the given name."""
    name = (name or "").lower()
    if name in ("anthropic", "claude"):
        return AnthropicProvider(settings)
    if name in ("mock", "offline", "test"):
        return MockProvider(settings)
    raise ValueError(
        f"Unknown provider {name!r}. Available: 'anthropic', 'mock'."
    )


__all__ = ["Provider", "AnthropicProvider", "MockProvider", "get_provider"]
