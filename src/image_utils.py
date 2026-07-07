"""Image loading, resizing, and encoding utilities.

Turns a file path, URL, raw bytes, or PIL image into the base64 payload a
vision LLM expects, resizing over-large images so requests stay small.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image as PILImage

# Map Pillow format names to the media types the API expects.
_MEDIA_TYPES = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "GIF": "image/gif",
    "WEBP": "image/webp",
}


@dataclass
class Image:
    """A normalized image ready to send to a vision model."""

    pil: PILImage.Image
    source: str = "<memory>"

    # ---- constructors -------------------------------------------------

    @classmethod
    def open(cls, path: str | Path) -> "Image":
        """Load an image from a local file path."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No such image file: {path}")
        img = PILImage.open(path)
        img.load()
        return cls(pil=img, source=str(path))

    @classmethod
    def from_bytes(cls, data: bytes, source: str = "<bytes>") -> "Image":
        img = PILImage.open(io.BytesIO(data))
        img.load()
        return cls(pil=img, source=source)

    @classmethod
    def from_url(cls, url: str, timeout: int = 30) -> "Image":
        """Download an image from a URL (requires the ``requests`` package)."""
        try:
            import requests
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Loading images from URLs needs the 'requests' package: "
                "pip install requests"
            ) from exc
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return cls.from_bytes(resp.content, source=url)

    # ---- transforms ---------------------------------------------------

    def resized(self, max_edge: int) -> "Image":
        """Return a copy whose longest edge is at most ``max_edge`` px."""
        w, h = self.pil.size
        longest = max(w, h)
        if longest <= max_edge:
            return self
        scale = max_edge / longest
        new_size = (round(w * scale), round(h * scale))
        return Image(pil=self.pil.resize(new_size, PILImage.LANCZOS),
                     source=self.source)

    # ---- encoding -----------------------------------------------------

    def to_base64(self, max_edge: int | None = None) -> tuple[str, str]:
        """Return ``(base64_data, media_type)`` for this image.

        Images are re-encoded as JPEG (or PNG if they have transparency) so
        the output is always a format the API accepts, regardless of input.
        """
        img = self.resized(max_edge).pil if max_edge else self.pil

        has_alpha = img.mode in ("RGBA", "LA", "P")
        buffer = io.BytesIO()
        if has_alpha:
            img.convert("RGBA").save(buffer, format="PNG")
            media_type = "image/png"
        else:
            img.convert("RGB").save(buffer, format="JPEG", quality=90)
            media_type = "image/jpeg"

        data = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")
        return data, media_type

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        w, h = self.pil.size
        return f"Image(source={self.source!r}, size={w}x{h}, mode={self.pil.mode})"
