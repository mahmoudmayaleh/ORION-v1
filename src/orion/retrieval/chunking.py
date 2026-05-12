"""Chunking and preprocessing for 3GPP/ETSI standards documents.

Note: For bulk ingestion of 3GPP specs, the TSpec-LLM HuggingFace dataset
(https://huggingface.co/datasets/hf-tspec-llm) ships pre-cleaned chunks
and is the recommended path. This chunker handles ad-hoc documents not
available in TSpec-LLM.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime

from .types import MemoryEntry

# 3GPP boilerplate patterns to strip
_BOILERPLATE_PATTERNS = [
    re.compile(r"^\s*3GPP\s+\d{4}\s*$", re.MULTILINE),
    re.compile(r"^\s*Change\s+history.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*Table\s+of\s+Contents.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*Copyright.*3GPP.*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*Figure\s+\d+[\.\-].*$", re.MULTILINE),
]

# 3GPP section heading pattern (e.g. "5.15.2.3 Title")
_SECTION_HEADING = re.compile(r"^(\d+(?:\.\d+)+)\s+(.+)$", re.MULTILINE)

# Equation block detector (lines with mostly math symbols)
_EQUATION_LINE = re.compile(r"^[\s=+\-*/^(){}\[\]|∑∏∫≤≥<>αβγδεφλμσρτ0-9.,]+$")


class StandardsCleaner:
    """Preprocessor for 3GPP/ETSI documents."""

    def clean(self, text: str) -> str:
        """Remove boilerplate, figure captions, and equation blocks."""
        for pattern in _BOILERPLATE_PATTERNS:
            text = pattern.sub("", text)

        # Remove pure equation blocks (3+ consecutive equation lines)
        lines = text.split("\n")
        cleaned_lines = []
        eq_buffer: list[str] = []

        for line in lines:
            if _EQUATION_LINE.match(line) and line.strip():
                eq_buffer.append(line)
            else:
                if len(eq_buffer) < 3:
                    cleaned_lines.extend(eq_buffer)
                eq_buffer = []
                cleaned_lines.append(line)

        if len(eq_buffer) < 3:
            cleaned_lines.extend(eq_buffer)

        return "\n".join(cleaned_lines)

    def extract_sections(self, text: str) -> list[tuple[str, str, str]]:
        """Extract (section_number, title, body) tuples from cleaned text."""
        sections: list[tuple[str, str, str]] = []
        matches = list(_SECTION_HEADING.finditer(text))

        for i, match in enumerate(matches):
            section_num = match.group(1)
            title = match.group(2).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            sections.append((section_num, title, body))

        return sections


def chunk_standards_document(
    text: str,
    source_doc: str,
    max_tokens: int = 250,
) -> list[MemoryEntry]:
    """Chunk a 3GPP/ETSI document into MemoryEntry objects.

    Args:
        text: Raw document text.
        source_doc: Document identifier (e.g. "TS 23.501 v17.6.0").
        max_tokens: Approximate max tokens per chunk (word-based estimate).

    Returns:
        List of MemoryEntry objects with appropriate tags.
    """
    cleaner = StandardsCleaner()
    cleaned = cleaner.clean(text)
    sections = cleaner.extract_sections(cleaned)
    entries: list[MemoryEntry] = []
    now = datetime.now()

    for section_num, title, body in sections:
        if not body.strip():
            continue

        # Split long sections into chunks
        words = body.split()
        chunks = []
        for i in range(0, len(words), max_tokens):
            chunk_words = words[i : i + max_tokens]
            chunks.append(" ".join(chunk_words))

        for chunk_idx, chunk in enumerate(chunks):
            entry_id = f"{source_doc}:{section_num}:{chunk_idx}"
            entry = MemoryEntry(
                entry_id=entry_id,
                topic=f"{title} ({section_num})",
                content=chunk,
                tags={
                    "source_doc": [source_doc],
                    "source_section": [section_num],
                },
                created_at=now,
                last_accessed_at=now,
            )
            entries.append(entry)

    # Handle text without section headings
    if not sections and cleaned.strip():
        words = cleaned.split()
        for i in range(0, len(words), max_tokens):
            chunk = " ".join(words[i : i + max_tokens])
            entry = MemoryEntry(
                entry_id=f"{source_doc}:body:{i // max_tokens}",
                topic=source_doc,
                content=chunk,
                tags={"source_doc": [source_doc]},
                created_at=now,
                last_accessed_at=now,
            )
            entries.append(entry)

    return entries
