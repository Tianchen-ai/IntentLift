#!/usr/bin/env python3
"""Render the original manuscript PDF figures for the GitHub README."""
from pathlib import Path
import pymupdf


def main() -> None:
    root = Path(__file__).resolve().parent / "figures"
    for name in ("fig01_overview", "fig09_cross_target"):
        with pymupdf.open(root / f"{name}.pdf") as doc:
            assert len(doc) == 1
            doc[0].get_pixmap(matrix=pymupdf.Matrix(2, 2)).save(root / f"{name}.png")
        print(f"Rendered {name}.png from the original manuscript PDF")


if __name__ == "__main__":
    main()
