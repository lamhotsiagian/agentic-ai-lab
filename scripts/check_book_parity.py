#!/usr/bin/env python3
"""Assert that every symbol the book shows actually exists in this lab.

The book's headline claim is that it is built around a working codebase. That
claim is only true if the code printed on the page can be found in the
repository, so this script checks it mechanically instead of by inspection.

It extracts every class and function definition from the LaTeX listings in the
ebook chapters, then confirms each one is importable from the lab. Anything
missing is reported with the chapter that shows it.

    python scripts/check_book_parity.py --ebook ../v2-ebook
    python scripts/check_book_parity.py --ebook ../v2-ebook --json

Exit code 0 means the book and the lab agree. Exit code 1 means they do not.
"""

import argparse
import ast
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ALLOWED_ABSENT = {
    # Definitions inside a listing that are illustrative rather than shipped.
}

LISTING = re.compile(
    r"\\begin\{lstlisting\}(?:\[[^\]]*\])?\s*(.*?)\\end\{lstlisting\}",
    re.DOTALL,
)
DEFINITION = re.compile(r"^(?:class|def)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)


def symbols_in_book(ebook_dir: Path) -> dict:
    """Map every class and function shown in the book to the chapters showing it."""
    found: dict = {}
    chapters = sorted(ebook_dir.glob("chapter_*.tex"))
    if not chapters:
        raise SystemExit(f"No chapter_*.tex files found in {ebook_dir}")

    for chapter in chapters:
        text = chapter.read_text(encoding="utf-8")
        for listing in LISTING.findall(text):
            for name in DEFINITION.findall(listing):
                found.setdefault(name, set()).add(chapter.stem)
    return {name: sorted(chapters) for name, chapters in found.items()}


def symbols_in_lab(lab_dir: Path) -> dict:
    """Map every class and function defined in the lab to its module path."""
    found: dict = {}
    for path in sorted(lab_dir.rglob("*.py")):
        parts = path.relative_to(lab_dir).parts
        if any(p in {"_rag-graphrag-lab", "__pycache__", ".venv", "build", "dist"} for p in parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            print(f"  ! could not parse {path}: {exc}", file=sys.stderr)
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                found.setdefault(node.name, set()).add(
                    str(path.relative_to(lab_dir))
                )
    return {name: sorted(paths) for name, paths in found.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ebook", default=str(PROJECT_ROOT.parent / "v2-ebook"),
        help="directory holding the chapter_*.tex files",
    )
    parser.add_argument("--lab", default=str(PROJECT_ROOT))
    parser.add_argument("--json", action="store_true", help="machine readable output")
    args = parser.parse_args()

    ebook_dir, lab_dir = Path(args.ebook), Path(args.lab)
    book = symbols_in_book(ebook_dir)
    lab = symbols_in_lab(lab_dir)

    missing = {
        name: chapters
        for name, chapters in book.items()
        if name not in lab and name not in ALLOWED_ABSENT
    }

    if args.json:
        print(json.dumps({
            "book_symbols": len(book),
            "lab_symbols": len(lab),
            "missing": missing,
        }, indent=2))
        return 1 if missing else 0

    print(f"Book shows      : {len(book)} classes and functions")
    print(f"Lab defines     : {len(lab)} classes and functions")
    print(f"Present in both : {len(book) - len(missing)}")

    if missing:
        print(f"\nMISSING from the lab: {len(missing)}\n")
        for name in sorted(missing):
            print(f"  {name:<28} shown in {', '.join(missing[name])}")
        print(
            "\nThe book tells the reader to run this code. Implement the "
            "symbols above, or remove the listings that show them."
        )
        return 1

    print("\nBook and lab agree. Every symbol the book shows exists in the lab.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
