import json
import os
import sys
from pathlib import Path

from .models import translate_text


def main():
    data = json.load(sys.stdin)
    root = Path(__file__).resolve().parents[2]
    model = os.getenv("BHARATDOC_TRANSLATION", str(root / "models" / "nllb"))
    print(translate_text(model, data["text"], data["source"], data["target"]))


if __name__ == "__main__":
    main()
