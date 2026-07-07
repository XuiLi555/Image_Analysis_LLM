"""Offline mock provider.

Returns deterministic, canned responses derived from the image's basic
properties — no network, no API key. This keeps the test suite and the demo
runnable anywhere (including CI) and gives newcomers something that "works"
before they add credentials.

It is intentionally NOT a real model: it only inspects size/colour, so its
"analysis" is a stand-in you can compare against the real provider.
"""

from __future__ import annotations

import json

from .base import Provider
from ..image_utils import Image


def _dominant_color(image: Image) -> str:
    small = image.pil.convert("RGB").resize((1, 1))
    r, g, b = small.getpixel((0, 0))
    if max(r, g, b) - min(r, g, b) < 20:
        return "grey" if r < 200 else "white"
    if r >= g and r >= b:
        return "red"
    if g >= r and g >= b:
        return "green"
    return "blue"


class MockProvider(Provider):
    """Deterministic offline provider used for tests and demos."""

    def analyze(self, prompt, images, system=None) -> str:
        img = images[0]
        w, h = img.pil.size
        color = _dominant_color(img)
        p = prompt.lower()

        if "json" in p or "classify" in p:
            return json.dumps(
                {"label": color, "confidence": 0.42, "note": "mock response"}
            )
        if "text" in p and ("read" in p or "extract" in p):
            return "[mock] no text detected in this image"
        if "compare" in p or len(images) > 1:
            return (
                f"[mock] Received {len(images)} images. They differ in size and "
                f"colour; a real model would describe specifics."
            )
        if "?" in prompt:
            return (
                f"[mock] Answering '{prompt.strip()}': this is a "
                f"{w}x{h} predominantly {color} image."
            )
        return (
            f"[mock] A {w}x{h} image whose dominant colour is {color}. "
            f"Set ANTHROPIC_API_KEY and use --provider anthropic for real "
            f"analysis."
        )
