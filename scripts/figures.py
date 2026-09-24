"""Write outputs/figures.json from the outputs, and say which prose it contradicts.

    python scripts/figures.py            # rewrite figures.json, then check the docs
    python scripts/figures.py --check    # exit 1 if figures.json or any doc is stale

Run it after any script that regenerates a published output. The prose check is
the same one ``tests/test_figures.py`` runs in CI; this runs it on a machine that
also holds the uncommitted outputs, which is the only place figures.json itself
can be checked.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from driftloop import figures

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = REPO_ROOT / "outputs"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="change nothing; exit 1 if stale")
    args = parser.parse_args()

    current = figures.collect(OUTPUTS)
    target = OUTPUTS / figures.FIGURES_FILE
    stale = not target.exists() or figures.load(OUTPUTS) != current
    if args.check:
        if stale:
            print(f"{target.relative_to(REPO_ROOT)} is stale: run python scripts/figures.py")
    else:
        target.write_text(json.dumps(current, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {len(current)} figures to {target.relative_to(REPO_ROOT)}")
        stale = False

    failed = stale
    for path in figures.documents(REPO_ROOT):
        text = path.read_text(encoding="utf-8")
        for problem in figures.check_text(text, current):
            print(f"{path.relative_to(REPO_ROOT)}: {problem}")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
