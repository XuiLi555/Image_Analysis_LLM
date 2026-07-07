"""High-level image-analysis API.

``ImageAnalyzer`` wraps a provider and exposes task-oriented methods. Each
method just builds a good prompt and delegates to the provider, so the prompt
engineering lives in one readable place.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import Settings
from .image_utils import Image
from .providers import get_provider

_SYSTEM = (
    "You are a precise visual analysis assistant. Base every statement only on "
    "what is visibly present in the image. If something is unclear or not "
    "shown, say so rather than guessing."
)


def _as_image(image) -> Image:
    """Accept an Image, a path, or raw bytes and return an Image."""
    if isinstance(image, Image):
        return image
    if isinstance(image, (bytes, bytearray)):
        return Image.from_bytes(bytes(image))
    if isinstance(image, (str, Path)):
        s = str(image)
        if s.startswith("http://") or s.startswith("https://"):
            return Image.from_url(s)
        return Image.open(s)
    raise TypeError(f"Cannot interpret {type(image)!r} as an image.")


class ImageAnalyzer:
    """Task-oriented interface over a vision-LLM provider."""

    def __init__(self, settings: Settings | None = None, provider: str | None = None):
        self.settings = settings or Settings.from_env()
        if provider:
            self.settings.provider = provider
        self.provider = get_provider(self.settings.provider, self.settings)

    # ---- core tasks ---------------------------------------------------

    def describe(self, image, detail: str = "detailed") -> str:
        """Describe / caption an image.

        ``detail`` is ``"brief"`` (one sentence) or ``"detailed"`` (a full
        paragraph covering subjects, setting, colours, and mood).
        """
        img = _as_image(image)
        if detail == "brief":
            prompt = "Describe this image in a single concise sentence."
        else:
            prompt = (
                "Describe this image in detail: the main subjects, the setting, "
                "notable colours, and the overall mood or context."
            )
        return self.provider.analyze(prompt, [img], system=_SYSTEM)

    def ask(self, image, question: str) -> str:
        """Answer a natural-language question about the image (VQA)."""
        img = _as_image(image)
        prompt = (
            f"Answer this question about the image as accurately as you can. "
            f"If the image doesn't contain enough information, say so.\n\n"
            f"Question: {question}"
        )
        return self.provider.analyze(prompt, [img], system=_SYSTEM)

    def extract_text(self, image) -> str:
        """Transcribe text visible in the image (OCR-style)."""
        img = _as_image(image)
        prompt = (
            "Read and transcribe all text visible in this image, preserving "
            "reading order and line breaks. If there is no text, reply exactly: "
            "NO TEXT FOUND."
        )
        return self.provider.analyze(prompt, [img], system=_SYSTEM)

    def classify(self, image, labels: list[str]) -> dict:
        """Classify the image into one of ``labels``. Returns parsed JSON.

        The returned dict has keys ``label``, ``confidence`` (0-1), and
        ``reason``. Falls back to raw text under ``_raw`` if parsing fails.
        """
        img = _as_image(image)
        label_list = ", ".join(labels)
        prompt = (
            f"Classify this image into exactly one of these categories: "
            f"{label_list}.\n"
            "Respond ONLY with a JSON object and nothing else, using this "
            'schema: {"label": <one of the categories>, '
            '"confidence": <number between 0 and 1>, '
            '"reason": <short justification>}.'
        )
        raw = self.provider.analyze(prompt, [img], system=_SYSTEM)
        return _parse_json(raw)

    def detect(self, image, objects: str) -> str:
        """Find and count described objects in the image."""
        img = _as_image(image)
        prompt = (
            f"Look for the following in the image: {objects}. "
            "For each, state whether it is present, roughly how many there are, "
            "and where it appears (e.g. top-left, centre). If absent, say so."
        )
        return self.provider.analyze(prompt, [img], system=_SYSTEM)

    def compare(self, image_a, image_b) -> str:
        """Compare two images: key similarities and differences."""
        a, b = _as_image(image_a), _as_image(image_b)
        prompt = (
            "You are shown two images, in order (first, then second). "
            "Summarise their main similarities, then their main differences."
        )
        return self.provider.analyze(prompt, [a, b], system=_SYSTEM)


def _parse_json(text: str) -> dict:
    """Best-effort JSON parse: strip code fences, find the outermost braces."""
    cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                pass
    return {"_raw": text}
