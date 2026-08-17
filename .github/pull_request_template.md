## What this changes

<!-- One or two sentences. If it changes a behaviour, say which. -->

## Why

<!-- What evidence motivated it. If it changes a choice the code comments justify with a
measurement or an incident, say what changed about that evidence. -->

## Checks

- [ ] `PYTHONHASHSEED=0 pytest` passes
- [ ] `ruff check . --select E9,F63,F7,F82,F811` passes
- [ ] A test covers the changed behaviour. If a guard changed, a test covers what it still refuses.
- [ ] No unrelated reformatting
- [ ] No frozen input changed in place (calibrated levels, benchmark, recorded references)
