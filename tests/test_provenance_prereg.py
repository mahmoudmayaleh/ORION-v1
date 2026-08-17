"""The `--no-prereg` hatch must relax exactly one thing and nothing else.

The pre-registrations are not distributed with the public repository, so without
a hatch every entry point refuses on a fresh clone. A hatch is also how a guard
gets lost, so its scope is pinned here rather than left to the docstring: it
applies when the document is ABSENT, and a document that is present is verified
in full whether or not the flag was passed.
"""

from __future__ import annotations

import subprocess

import pytest

from orion import provenance as P


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=str(repo), check=True,
                   capture_output=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A throwaway git repo standing in for the checkout.

    `core.autocrlf` is pinned off. The check under test compares the working
    tree against the committed blob byte for byte, so on a machine with
    autocrlf on, git would rewrite line endings on the way into the blob and
    every committed pre-registration would read as drifted. The test would then
    pass or fail on git config rather than on the code.
    """
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "config", "user.name", "test")
    _git(tmp_path, "config", "core.autocrlf", "false")
    (tmp_path / "docs").mkdir()
    (tmp_path / "seed.txt").write_bytes(b"seed\n")
    _git(tmp_path, "add", "seed.txt")
    _git(tmp_path, "commit", "-q", "-m", "seed")
    monkeypatch.setattr(P, "_REPO", tmp_path)
    return tmp_path


def test_absent_prereg_refuses_by_default(repo):
    with pytest.raises(P.ProvenanceError, match="not found"):
        P.verify_prereg("docs/PREREG_ABSENT.md")


def test_absent_prereg_is_skipped_and_labelled_when_allowed(repo):
    block = P.verify_prereg("docs/PREREG_ABSENT.md", allow_absent=True)
    assert block["status"] == "skipped"
    assert block["sha256"] is None
    assert block["committed_sha256"] is None
    # The skip must be visible in whatever the runner stamps into its result.
    assert "not present" in block["reason"]


def test_present_but_uncommitted_still_refuses_under_the_hatch(repo):
    """The hatch must not become a way to run against an unpinned document."""
    (repo / "docs" / "PREREG_X.md").write_text("draft\n", encoding="utf-8")
    with pytest.raises(P.ProvenanceError, match="not committed at HEAD"):
        P.verify_prereg("docs/PREREG_X.md", allow_absent=True)


def test_present_and_drifted_still_refuses_under_the_hatch(repo):
    (repo / "docs" / "PREREG_X.md").write_text("v1\n", encoding="utf-8")
    _git(repo, "add", "docs/PREREG_X.md")
    _git(repo, "commit", "-q", "-m", "prereg")
    (repo / "docs" / "PREREG_X.md").write_text("v2 edited after the run\n",
                                               encoding="utf-8")
    with pytest.raises(P.ProvenanceError, match="drifted"):
        P.verify_prereg("docs/PREREG_X.md", allow_absent=True)


def test_committed_and_clean_verifies_under_either_setting(repo):
    (repo / "docs" / "PREREG_X.md").write_text("frozen\n", encoding="utf-8")
    _git(repo, "add", "docs/PREREG_X.md")
    _git(repo, "commit", "-q", "-m", "prereg")
    for allow in (False, True):
        block = P.verify_prereg("docs/PREREG_X.md", allow_absent=allow)
        assert block["status"] == "verified"
        assert block["sha256"] == block["committed_sha256"]


def test_cited_hash_mismatch_still_refuses_under_the_hatch(repo):
    (repo / "docs" / "PREREG_X.md").write_text("frozen\n", encoding="utf-8")
    _git(repo, "add", "docs/PREREG_X.md")
    _git(repo, "commit", "-q", "-m", "prereg")
    with pytest.raises(P.ProvenanceError, match="does not match the cited"):
        P.verify_prereg("docs/PREREG_X.md", cited_sha256="deadbeef",
                        allow_absent=True)
