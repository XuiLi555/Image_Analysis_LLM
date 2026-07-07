"""End-to-end ImageLens demo.

Runs every analysis task against the bundled sample images.

By default it uses the OFFLINE mock provider, so it works with no API key:

    python examples/demo.py

To use the real Claude vision model, set your key and pick the provider:

    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/demo.py --provider anthropic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imagelens import ImageAnalyzer  # noqa: E402

SAMPLES = ROOT / "examples" / "sample_images"


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="ImageLens demo")
    parser.add_argument(
        "--provider",
        default="mock",
        help="'mock' (offline, default) or 'anthropic' (needs ANTHROPIC_API_KEY)",
    )
    args = parser.parse_args()

    analyzer = ImageAnalyzer(provider=args.provider)
    print(f"Using provider: {analyzer.settings.provider} "
          f"(model: {analyzer.settings.model})")

    shapes = SAMPLES / "shapes.png"
    blue_box = SAMPLES / "blue_box.png"

    section("1. Describe (detailed)")
    print(analyzer.describe(shapes))

    section("2. Describe (brief)")
    print(analyzer.describe(shapes, detail="brief"))

    section("3. Visual question answering")
    print(analyzer.ask(shapes, "What shape is in the middle and what colour is it?"))

    section("4. Extract text")
    print(analyzer.extract_text(shapes))

    section("5. Classify")
    print(analyzer.classify(shapes, labels=["landscape", "diagram", "portrait", "abstract"]))

    section("6. Detect objects")
    print(analyzer.detect(shapes, objects="a circle, a square, any text"))

    section("7. Compare two images")
    print(analyzer.compare(shapes, blue_box))

    print("\nDone.\n")


if __name__ == "__main__":
    main()
