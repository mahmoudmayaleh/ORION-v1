"""Every script under scripts/ must parse.

Commit 65b3d51 (the provenance-guard patch) inserted an import into the middle of a
parenthesized import in scripts/rc_train_runner.py, leaving the Track D training
entry point a hard SyntaxError. It survived a day and two commits because no test
imports these scripts and nothing ran that file in between. Parsing is the cheapest
possible check and it would have caught it instantly.
"""
import py_compile
from pathlib import Path

import pytest

SCRIPTS = sorted((Path(__file__).resolve().parent.parent / "scripts").glob("*.py"))


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_script_compiles(path):
    try:
        py_compile.compile(str(path), doraise=True)
    except py_compile.PyCompileError as e:
        pytest.fail(f"{path.name} does not parse:\n{e}")
