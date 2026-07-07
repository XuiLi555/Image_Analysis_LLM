"""ImageLens — analyze images with a vision LLM.

A small, provider-agnostic toolkit that sends images to a multimodal large
language model and returns natural-language analysis. High-level tasks:

    - describe(image)            -> caption / detailed description
    - ask(image, question)       -> visual question answering (VQA)
    - extract_text(image)        -> read text in the image (OCR-style)
    - classify(image, labels)    -> pick the best label(s), structured JSON
    - detect(image, objects)     -> locate/count described objects
    - compare(image_a, image_b)  -> describe differences & similarities

The default backend is the Anthropic Claude API, but any backend implementing
the ``Provider`` interface can be plugged in. A built-in ``MockProvider`` lets
the tests and demo run completely offline with no API key.
"""

from .analyzer import ImageAnalyzer
from .image_utils import Image

__version__ = "0.1.0"

__all__ = ["ImageAnalyzer", "Image", "__version__"]
