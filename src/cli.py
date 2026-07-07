"""Command-line interface for ImageLens.

Examples
--------
    imagelens describe photo.jpg
    imagelens describe photo.jpg --detail brief
    imagelens ask photo.jpg "How many people are in this picture?"
    imagelens text receipt.png
    imagelens classify animal.jpg --labels cat dog bird
    imagelens detect street.jpg --objects "traffic lights, pedestrians"
    imagelens compare before.jpg after.jpg
    imagelens describe photo.jpg --provider mock      # offline, no API key

The console script ``imagelens`` is installed via pyproject; you can also run
``python -m imagelens.cli ...`` without installing.
"""

from __future__ import annotations

import argparse
import json
import sys

from .analyzer import ImageAnalyzer
from .config import Settings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="imagelens", description="Analyze images with a vision LLM."
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Backend to use: 'anthropic' (default) or 'mock' (offline).",
    )
    parser.add_argument(
        "--model", default=None, help="Override the model name."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("describe", help="Describe/caption an image")
    p.add_argument("image")
    p.add_argument("--detail", choices=["brief", "detailed"], default="detailed")

    p = sub.add_parser("ask", help="Ask a question about an image")
    p.add_argument("image")
    p.add_argument("question")

    p = sub.add_parser("text", help="Extract text from an image")
    p.add_argument("image")

    p = sub.add_parser("classify", help="Classify an image into labels")
    p.add_argument("image")
    p.add_argument("--labels", nargs="+", required=True)

    p = sub.add_parser("detect", help="Detect/count objects in an image")
    p.add_argument("image")
    p.add_argument("--objects", required=True, help="Comma-separated things to find")

    p = sub.add_parser("compare", help="Compare two images")
    p.add_argument("image_a")
    p.add_argument("image_b")

    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    settings = Settings.from_env()
    if args.model:
        settings.model = args.model

    try:
        analyzer = ImageAnalyzer(settings=settings, provider=args.provider)

        if args.command == "describe":
            print(analyzer.describe(args.image, detail=args.detail))
        elif args.command == "ask":
            print(analyzer.ask(args.image, args.question))
        elif args.command == "text":
            print(analyzer.extract_text(args.image))
        elif args.command == "classify":
            print(json.dumps(analyzer.classify(args.image, args.labels), indent=2))
        elif args.command == "detect":
            print(analyzer.detect(args.image, args.objects))
        elif args.command == "compare":
            print(analyzer.compare(args.image_a, args.image_b))
    except (RuntimeError, ImportError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
