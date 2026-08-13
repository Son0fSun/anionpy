"""Path bootstrap for the differential suite.

Every module in this package that needs `import anionpy` imports this module
first. It makes the repo root importable no matter how the suite is
invoked: `pytest` from the repo root, `pytest tests/differential/`, or
`python3 tests/differential/run.py` run directly (which only puts this
directory, not the repo root, on sys.path).

No numerical work happens here -- this is import plumbing only.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent

for _p in (str(REPO_ROOT), str(THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
