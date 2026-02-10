#!/usr/bin/env python3
"""
Update design_github.md from design.md for GitHub rendering.
Escapes underscores in math, collapses multi-line $$ blocks to single line,
copies ieee14_diagram.png to documents/images and uses path images/ieee14_diagram.png.
Do not create any README or readme_design_github in documents/images.

Run from project root: python3 documents/update_design_for_github.py
"""

import re
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DESIGN_MD = SCRIPT_DIR / "design.md"
DESIGN_GITHUB_MD = SCRIPT_DIR / "design_github.md"
IMAGES_DIR = SCRIPT_DIR / "images"
SOURCE_DIAGRAM = PROJECT_ROOT / "tests" / "output" / "ieee14_diagram.png"
DIAGRAM_NAME = "ieee14_diagram.png"
IMAGE_PATH_IN_DOC = "images/ieee14_diagram.png"
OLD_IMAGE_PATTERN = re.compile(r"\.\./tests/output/ieee14_diagram\.png", re.IGNORECASE)


def escape_underscores_in_math(text: str) -> str:
    return text.replace("_", "\\_")


def process_display_math(content: str) -> str:
    def replace_block(m: re.Match) -> str:
        inner = m.group(1)
        inner = escape_underscores_in_math(inner)
        inner = " ".join(inner.split())
        return "$$ " + inner + " $$"
    return re.sub(r"\$\$\s*([\s\S]*?)\s*\$\$", replace_block, content)


def process_inline_math(content: str) -> str:
    def replace_inline(m: re.Match) -> str:
        inner = m.group(1)
        inner = escape_underscores_in_math(inner)
        return "$" + inner + "$"
    return re.sub(r"(?<!\$)\$([^$\n]+?)\$(?!\$)", replace_inline, content)


def ensure_image_synced() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    if SOURCE_DIAGRAM.exists():
        shutil.copy2(SOURCE_DIAGRAM, IMAGES_DIR / DIAGRAM_NAME)
    else:
        raise FileNotFoundError("Source diagram not found: %s" % SOURCE_DIAGRAM)


def update_image_path(text: str) -> str:
    return OLD_IMAGE_PATTERN.sub(IMAGE_PATH_IN_DOC, text)


def main() -> None:
    ensure_image_synced()
    text = DESIGN_MD.read_text(encoding="utf-8")
    text = process_display_math(text)
    text = process_inline_math(text)
    text = update_image_path(text)
    lines = text.split("\n")
    if len(lines) >= 2:
        head = [lines[0], "", "*GitHub version. Keep in sync with design.md. Run: `python3 documents/update_design_for_github.py`*", ""]
        body = lines[2:] if lines[1].strip() == "" else lines[1:]
        out_lines = head + body
    else:
        out_lines = lines
    output = "\n".join(out_lines)
    DESIGN_GITHUB_MD.write_text(output, encoding="utf-8")
    print("Updated design_github.md and synced images/ieee14_diagram.png.")


if __name__ == "__main__":
    main()
