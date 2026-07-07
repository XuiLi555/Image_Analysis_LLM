"""Anthropic Claude vision provider.

Sends images to the Claude Messages API. Each image becomes an ``image``
content block (base64) followed by the text prompt. The ``anthropic`` SDK is
imported lazily so the package imports fine without it — you only need it to
actually call the API.

Docs: https://docs.claude.com/en/docs/build-with-claude/vision
"""

from __future__ import annotations

from .base import Provider
from ..image_utils import Image


class AnthropicProvider(Provider):
    """Calls Claude's multimodal Messages API."""

    def __init__(self, settings):
        super().__init__(settings)
        self._client = None  # created on first use

    def _client_or_raise(self):
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The Anthropic provider needs the 'anthropic' package: "
                "pip install anthropic"
            ) from exc

        if not self.settings.api_key:
            raise RuntimeError(
                "No API key found. Set the ANTHROPIC_API_KEY environment "
                "variable (or copy .env.example to .env and fill it in)."
            )
        self._client = anthropic.Anthropic(api_key=self.settings.api_key)
        return self._client

    def _image_block(self, image: Image) -> dict:
        data, media_type = image.to_base64(max_edge=self.settings.max_image_edge)
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": data,
            },
        }

    def analyze(self, prompt, images, system=None) -> str:
        client = self._client_or_raise()

        content = [self._image_block(img) for img in images]
        content.append({"type": "text", "text": prompt})

        kwargs = {
            "model": self.settings.model,
            "max_tokens": self.settings.max_tokens,
            "messages": [{"role": "user", "content": content}],
        }
        if system:
            kwargs["system"] = system

        response = client.messages.create(**kwargs)

        # Concatenate all text blocks in the response.
        return "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
