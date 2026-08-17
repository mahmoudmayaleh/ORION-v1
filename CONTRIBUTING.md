# Contributing

This repository is the software artifact for a research paper. Issues and pull requests
are welcome, particularly ones that report a defect, a result that does not reproduce, or
an unclear part of the setup.

## Reporting a problem

Open an issue with the command you ran, the full output, your Python version and operating
system, and whether a language-model server was involved. If a run produced a result file,
its `provenance` block names the commit and the serving model, and pasting it identifies
the tree exactly.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,actors,retrieval]"
PYTHONHASHSEED=0 pytest
```

`PYTHONHASHSEED=0` is required. Behaviour cloning replays a greedy placement whose
tie-breaking reads set iteration order, so without it the demonstrations are not
reproducible and unrelated tests fail intermittently.

## Pull requests

- Keep the test suite green. `PYTHONHASHSEED=0 pytest`
- Keep the defect lint gate green. `ruff check . --select E9,F63,F7,F82,F811`
- Add a test for behaviour you change. A change to a guard needs a test that the guard
  still refuses what it is supposed to refuse, not only that it permits what it should.
- Explain why in the code, not only what. Much of the commentary in this codebase records
  a measurement or an incident that motivated a choice. If you change such a choice, say
  what evidence changed.
- Do not reformat unrelated code. Result files stamp the commit they ran under, so a bulk
  reformat makes those stamps point at a tree that differs from the one that produced the
  numbers.

## Things that will be refused

- Weakening `orion.provenance`. The untracked-code refusal has no bypass by design. A run
  whose code exists in no revision can never be reproduced or refuted.
- Changing a frozen input. The calibrated load ladder, the frozen benchmark and the
  recorded reference values are fixed so that results stay comparable across runs. If one
  is wrong, the fix is a new frozen value with the reason recorded, not an edit in place.

## Scope

The experiment protocol and its pre-registrations are not distributed here. Changes that
depend on them, such as adding an approach to the reported ladder, are outside what can be
reviewed against this repository alone.
