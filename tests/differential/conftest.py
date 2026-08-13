"""pytest entry point for this directory. This directory has no __init__.py
on purpose (see README.md), so pytest's default "prepend" import mode
already puts this directory on sys.path[0] when collecting files here --
but we pin it explicitly too, since relying on an implicit pytest import
mode default is exactly the kind of thing that quietly breaks when
someone runs pytest from a different cwd or with -p no:cacheprovider.
"""
import sys
from pathlib import Path

_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
