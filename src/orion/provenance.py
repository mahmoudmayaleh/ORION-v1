"""Run provenance: the single recorder every result-writing runner must pass through.

Why this is a hard guard and not a warning
------------------------------------------
The 2026-07-15 R.2|42 = 86.6% result is unfalsifiable. It does not reproduce on
identical inputs (12-20/100 on a server since proven healthy), and the code that
produced it was UNTRACKED, so there is no revision to diff against. The Delta-3
dirty-tree check did not catch it because it ran with
`git status --porcelain --untracked-files=no` -- untracked files were invisible
to it by construction.

An untracked runner is strictly worse than a dirty one: a dirty tree at least
records a base commit to diff from. An untracked file leaves no trace at all, so
a number it produced can never be reproduced or refuted. That is not a
provenance inconvenience; it is what turns a bad number into a permanent one.

So: any `??` under scripts/ or src/ REFUSES the run. There is deliberately no
environment-variable bypass -- a bypass is how the last guard was lost. Commit
the file (or add a genuine build artifact to .gitignore) and re-run.

The one narrow exception is `verify_prereg(allow_absent=True)`, reached only via
the runners' `--no-prereg` flag. The pre-registrations are not distributed with
this repository, so without it every entry point refuses on a fresh clone. It
applies only when the document is MISSING, it relaxes nothing when the document
is present, it does not touch the untracked-code refusal above, and any run that
uses it stamps `prereg.status = "skipped"` into its result JSON.

Usage
-----
    from orion.provenance import git_provenance
    prov = git_provenance()          # raises ProvenanceError on untracked code
    out["provenance"] = prov         # stamp into the result JSON

`serving=` carries the LLM serving block (model_id, chat_format, server_pid) so
a cell records what answered it, not just what ran it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent
# Directories whose contents are "code that can change a number". A stray
# untracked file here is a provenance hole; elsewhere (runs/, results/) it is an
# artifact and none of our business.
_CODE_DIRS = ("scripts/", "src/")


class ProvenanceError(RuntimeError):
    """Provenance cannot be established, so the run must not produce a number."""


def _git(*args: str) -> str:
    """Run git and FAIL CLOSED. The old inline helper returned None on any error,
    so `bool(_git([...]))` silently recorded dirty=False when git itself broke --
    a clean-looking provenance block for an unprovenanced run."""
    try:
        return subprocess.check_output(
            ["git", *args], cwd=str(_REPO), stderr=subprocess.PIPE
        ).decode().strip()
    except FileNotFoundError as e:
        raise ProvenanceError(f"git is not installed; cannot establish provenance: {e}") from e
    except subprocess.CalledProcessError as e:
        raise ProvenanceError(
            f"git {' '.join(args)} failed ({e.returncode}): {e.stderr.decode(errors='replace')[:200]}"
        ) from e


def untracked_code(porcelain: str | None = None) -> list[str]:
    """Paths of untracked files under scripts/ or src/ (the `??` entries)."""
    if porcelain is None:
        porcelain = _git("status", "--porcelain", "--untracked-files=all")
    out = []
    for line in porcelain.splitlines():
        if not line.startswith("??"):
            continue
        path = line[2:].strip().strip('"')
        if any(path.startswith(d) for d in _CODE_DIRS):
            out.append(path)
    return out


def verify_prereg(prereg_path: str | Path, cited_sha256: str | None = None,
                  allow_absent: bool = False) -> dict:
    """Verify a pre-registration against its COMMITTED copy. Refuses on drift.

    Hash-citing an untracked file is self-defeating: the hash proves the document
    did not change only if something independent pins what the document was. The
    §R amendment was edited at 18:04 on 2026-07-15, an hour after the R.1/R.2 run
    it governed (33fd3506fead -> f7ecd2219b79), while untracked. That edit was
    legitimate -- an appended, labelled "post R.1/R.2" section -- but nothing made
    it visible, and an illegitimate edit would have looked identical.

    So: the working-tree pre-reg must byte-match the copy committed at HEAD. Drift
    between docs/ and git is a refusal condition, not a warning.

    The pre-registrations are not distributed with this repository. `allow_absent`
    exists for that case and for that case only: it applies when the document is
    MISSING from the checkout, and it does not weaken any check that can still be
    made. If the document is present, verification proceeds in full and the flag
    changes nothing, so a working tree that has the documents cannot use this to
    dodge a drift refusal.

    A skipped verification is recorded, not hidden. The returned block carries
    `status="skipped"` and it is stamped into the result JSON, so a number produced
    without a pinned pre-registration says so on its face and can never be mistaken
    for a governed one.

    Args:
        prereg_path: path to the pre-registration (repo-relative or absolute).
        cited_sha256: optional hash the caller claims to be running under; a
            prefix is accepted (the record cites 12 chars). Mismatch refuses.
        allow_absent: return a `status="skipped"` block instead of refusing when
            the document is not in this checkout. Never relaxes anything else.

    Returns:
        {"path", "sha256", "committed_sha256", "matches_cited"}, or
        {"path", "status": "skipped", ...} when absent and allow_absent is set.
    """
    import hashlib

    path = Path(prereg_path)
    abs_path = path if path.is_absolute() else (_REPO / path)
    try:
        rel = str(abs_path.resolve().relative_to(_REPO.resolve())).replace("\\", "/")
    except ValueError as e:
        raise ProvenanceError(f"pre-registration {abs_path} is outside the repo") from e

    if not abs_path.exists():
        if not allow_absent:
            raise ProvenanceError(f"pre-registration not found: {rel}")
        import sys as _sys
        print(
            f"\n*** PRE-REGISTRATION NOT VERIFIED: {rel} is not in this checkout.\n"
            f"*** The run continues because --no-prereg was passed. Its result JSON\n"
            f"*** will record prereg.status = \"skipped\", and the number it produces\n"
            f"*** is NOT pinned to a pre-registered protocol.\n",
            file=_sys.stderr,
        )
        return {"path": rel, "status": "skipped",
                "reason": "not present in this checkout; --no-prereg was passed",
                "sha256": None, "committed_sha256": None,
                "matches_cited": None, "cited_sha256": cited_sha256}

    working = abs_path.read_bytes()
    working_sha = hashlib.sha256(working).hexdigest()

    # The committed blob at HEAD. `git show` fails if the path is not in HEAD,
    # which is itself the failure: an uncommitted pre-reg pins nothing.
    try:
        committed = subprocess.check_output(
            ["git", "show", f"HEAD:{rel}"], cwd=str(_REPO), stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        raise ProvenanceError(
            f"\n*** REFUSING TO RUN: pre-registration {rel} is not committed at HEAD.\n"
            f"*** A cited prereg_sha256 pins nothing if the document it names exists\n"
            f"*** only in the working tree. Commit it first.\n"
            f"*** git: {e.stderr.decode(errors='replace')[:160]}"
        ) from e

    committed_sha = hashlib.sha256(committed).hexdigest()
    if committed_sha != working_sha:
        raise ProvenanceError(
            f"\n*** REFUSING TO RUN: {rel} has drifted from its committed copy.\n"
            f"***   working tree: {working_sha[:12]}\n"
            f"***   committed   : {committed_sha[:12]}\n"
            f"*** The run would cite a hash for a document git does not have. Commit\n"
            f"*** the amendment (append and label it, per the Δ2-R precedent) and re-run."
        )

    matches_cited = None
    if cited_sha256:
        matches_cited = working_sha.startswith(cited_sha256.strip().lower())
        if not matches_cited:
            raise ProvenanceError(
                f"\n*** REFUSING TO RUN: {rel} does not match the cited pre-registration.\n"
                f"***   cited : {cited_sha256}\n"
                f"***   actual: {working_sha[:len(cited_sha256)]}\n"
                f"*** The run is not governed by the document it claims."
            )

    return {"path": rel, "status": "verified", "sha256": working_sha,
            "committed_sha256": committed_sha, "matches_cited": matches_cited}


def git_provenance(serving: dict | None = None, tag: str | None = None,
                   prereg: str | Path | None = None,
                   cited_prereg_sha256: str | None = None,
                   allow_absent_prereg: bool = False) -> dict:
    """Provenance block for a result JSON. Raises ProvenanceError on untracked code.

    Args:
        serving: optional LLM serving block (model_id, chat_format, server_pid).
        tag: optional run tag, recorded for cross-referencing.
        allow_absent_prereg: pass through to verify_prereg. Applies only when the
            pre-registration is missing from the checkout; see verify_prereg. The
            untracked-code refusal above is unaffected and still has no bypass.

    Returns:
        dict with git_commit, git_dirty, git_dirty_files, and (if given) serving/tag.
    """
    commit = _git("rev-parse", "HEAD")
    porcelain = _git("status", "--porcelain", "--untracked-files=all")

    stray = untracked_code(porcelain)
    if stray:
        listed = "\n".join(f"      ?? {p}" for p in stray[:20])
        more = f"\n      ... and {len(stray) - 20} more" if len(stray) > 20 else ""
        raise ProvenanceError(
            "\n*** REFUSING TO RUN: untracked code under scripts/ or src/.\n"
            "*** A number produced by untracked code can never be reproduced or\n"
            "*** refuted -- that is exactly how R.2|42 = 86.6% became unfalsifiable.\n"
            f"{listed}{more}\n"
            "*** Fix: `git add` the file(s) and commit, or add a genuine build\n"
            "*** artifact to .gitignore. There is no bypass flag by design."
        )

    # Tracked-but-modified is recorded, not refused: the base commit makes it
    # diffable, so the number stays falsifiable.
    dirty_files = [ln[3:].strip() for ln in porcelain.splitlines() if not ln.startswith("??")]
    prov = {
        "git_commit": commit,
        "git_dirty": bool(dirty_files),
        "git_dirty_files": dirty_files[:50],
        "provenance_guard": "untracked-code refusal active (orion.provenance)",
    }
    # A pre-reg that drifts from its committed copy refuses the run: the cited hash
    # must name a document git actually has.
    if prereg is not None:
        prov["prereg"] = verify_prereg(prereg, cited_prereg_sha256,
                                       allow_absent=allow_absent_prereg)
    if serving is not None:
        prov["serving"] = serving
    if tag is not None:
        prov["tag"] = tag
    return prov


def serving_provenance(port: int = 8000) -> dict:
    """Record the SERVING LAYER: model_id, chat_format, and server incarnation.

    The 2026-07-15 incident pinned weights and prompts but not the chat template
    or the server process between them. `server_start` matters most: the July-8
    incarnation that produced R.2's 86.6% could not be characterised afterwards
    because nothing recorded which process answered, and its log did not exist.
    Record the incarnation while it is still answering.
    """
    import json as _json
    import re as _re

    prov: dict = {"port": port, "model_id": None, "chat_format": None,
                  "server_pid": None, "server_start": None}
    try:
        import urllib.request
        with urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=5) as r:
            prov["model_id"] = _json.loads(r.read())["data"][0]["id"]
    except Exception:  # noqa: BLE001
        pass
    try:
        out = subprocess.run(["ps", "-eo", "pid,lstart,cmd"],
                             capture_output=True, text=True).stdout
        for line in out.splitlines():
            if "llama_cpp.server" in line and f"--port {port}" in line:
                parts = line.split()
                prov["server_pid"] = parts[0]
                prov["server_start"] = " ".join(parts[1:6])
                m = _re.search(r"--chat_format\s+(\S+)", line)
                prov["chat_format"] = m.group(1) if m else "(auto-detected)"
                break
    except Exception:  # noqa: BLE001
        pass
    return prov
