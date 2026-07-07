"""Tests that run fully offline via the mock provider.  Run:  pytest -q

No API key required — these exercise image handling, prompt routing, and the
analyzer surface without hitting any network.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imagelens import ImageAnalyzer, Image
from imagelens.analyzer import _parse_json

SAMPLES = Path(__file__).resolve().parents[1] / "examples" / "sample_images"
SHAPES = SAMPLES / "shapes.png"
BLUE = SAMPLES / "blue_box.png"


def _analyzer():
    return ImageAnalyzer(provider="mock")


def test_image_loads_and_resizes():
    img = Image.open(SHAPES)
    assert img.pil.size == (400, 300)
    small = img.resized(max_edge=100)
    assert max(small.pil.size) == 100


def test_image_to_base64():
    data, media_type = Image.open(SHAPES).to_base64(max_edge=200)
    assert media_type in ("image/jpeg", "image/png")
    assert isinstance(data, str) and len(data) > 100


def test_describe():
    out = _analyzer().describe(SHAPES)
    assert isinstance(out, str) and out


def test_ask_returns_answer():
    out = _analyzer().ask(SHAPES, "What is in the middle?")
    assert "?" not in out.split("]")[0]  # answered, not echoed as a bare question
    assert out


def test_extract_text():
    out = _analyzer().extract_text(SHAPES)
    assert isinstance(out, str)


def test_classify_returns_dict():
    result = _analyzer().classify(SHAPES, labels=["a", "b"])
    assert isinstance(result, dict)
    assert "label" in result or "_raw" in result


def test_compare_two_images():
    out = _analyzer().compare(SHAPES, BLUE)
    assert "2 images" in out or "two" in out.lower()


def test_parse_json_with_code_fence():
    parsed = _parse_json('```json\n{"label": "cat", "confidence": 0.9}\n```')
    assert parsed["label"] == "cat"


def test_parse_json_embedded():
    parsed = _parse_json('Sure! {"label": "dog"} hope that helps')
    assert parsed["label"] == "dog"


def test_accepts_bytes_input():
    data = SHAPES.read_bytes()
    out = _analyzer().describe(data)
    assert out
