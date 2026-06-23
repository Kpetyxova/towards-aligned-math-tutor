"""Restore compressed training-set JSON files.
"""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRAINING_SETS_DIR = ROOT / "data" / "training_sets"


def unpack_file(source: Path) -> Path:
    target = source.with_suffix("")
    if target.exists():
        print(f"skip {target.relative_to(ROOT)}")
        return target
    with gzip.open(source, "rb") as src, target.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    print(f"wrote {target.relative_to(ROOT)}")
    return target


def main() -> None:
    compressed_files = sorted(TRAINING_SETS_DIR.glob("*/*.json.gz"))
    if not compressed_files:
        raise SystemExit(f"No .json.gz files found under {TRAINING_SETS_DIR}")
    for source in compressed_files:
        unpack_file(source)


if __name__ == "__main__":
    main()
