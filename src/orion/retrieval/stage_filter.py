"""Stage 1: Metadata filtering — AND across tag keys, OR within values."""

from __future__ import annotations

from .types import MemoryEntry, RetrievalQuery


def apply_metadata_filter(
    entries: list[MemoryEntry], query: RetrievalQuery
) -> list[MemoryEntry]:
    """Filter entries by tag constraints from the query.

    Logic:
      - AND across different tag keys (all keys must match).
      - OR within values for a single key (any value matches).
      - If an entry is missing a tag key, it passes (wildcard).
    """
    if not query.filters:
        return entries

    result = []
    for entry in entries:
        if _matches(entry, query.filters):
            result.append(entry)
    return result


def _matches(entry: MemoryEntry, filters: dict[str, str | list[str]]) -> bool:
    """Check if an entry matches all filter constraints."""
    for key, required in filters.items():
        entry_values = entry.tags.get(key)
        if entry_values is None:
            # Missing tag = wildcard match
            continue

        if isinstance(required, str):
            required_set = {required}
        else:
            required_set = set(required)

        # OR within values: at least one entry value in required set
        if not any(v in required_set or v == "all" for v in entry_values):
            return False
    return True
