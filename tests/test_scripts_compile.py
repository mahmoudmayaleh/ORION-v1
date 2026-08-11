"""Every script under scripts/ must parse.

Commit 65b3d51 (the provenance-guard patch) inserted an import into the middle of a
parenthesized import in scripts/rc_train_runner.py, leaving the Track D training
entry point a hard SyntaxError. It survived a day and two commits because no test
imports these scripts and nothing ran that file in between. Parsing is the cheapest
possible check and it would have caught it instantly.
"""
import ast
import io
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


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_no_script_re_execs_on_import(path):
    """A PYTHONHASHSEED pin must sit behind __main__.

    `milp_approach_runner` held it at module level and `test_y_acceptance_and_timeout`
    imports that module, so `pytest tests/` replaced its own process via os.execv part
    way through: 29 of 31 tests ran, no summary printed, exit code 0. A green-looking
    truncated suite is worse than a red one. grid_runner.py:61 fixed this once already.
    """
    src = io.open(path, encoding="utf-8").read()
    if "PYTHONHASHSEED" not in src or "os.execv" not in src:
        return
    tree = ast.parse(src)
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        if "os.execv" not in ast.unparse(node):
            continue
        assert "__main__" in ast.unparse(node.test), (
            f"{path.name} re-execs on IMPORT; put the pin behind "
            f"`__name__ == \"__main__\"`")
