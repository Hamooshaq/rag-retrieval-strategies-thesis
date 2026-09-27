from __future__ import annotations

import re
from dataclasses import dataclass

from storage.models import ParsedBlock, RetrievalUnit, UnitLevel


@dataclass(frozen=True)
class LocatedToken:
    text: str
    locator: str


def _tokens(text: str) -> list[str]:
    return re.findall(r"\S+", text)


def _windows(tokens: list[LocatedToken], size: int, overlap: int):
    if size <= 0:
        raise ValueError("Chunk size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("Overlap must satisfy 0 <= overlap < size")
    step = size - overlap
    for start in range(0, len(tokens), step):
        window = tokens[start : start + size]
        if window:
            yield start, window
        if start + size >= len(tokens):
            break


def _ordered_unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def chunk_fixed(
    blocks: list[ParsedBlock], size: int, overlap: int
) -> list[RetrievalUnit]:
    located: list[LocatedToken] = []
    for block in sorted(blocks, key=lambda item: item.order):
        located.extend(LocatedToken(token, block.locator) for token in _tokens(block.text))
    if not located:
        raise ValueError("Cannot create fixed chunks from empty text")
    document_ids = {block.document_id for block in blocks}
    if len(document_ids) != 1:
        raise ValueError("chunk_fixed expects blocks from exactly one document")
    document_id = next(iter(document_ids))
    units: list[RetrievalUnit] = []
    for index, (start, window) in enumerate(_windows(located, size, overlap), 1):
        text = " ".join(token.text for token in window)
        units.append(
            RetrievalUnit(
                unit_id=f"{document_id}:fixed:{index}",
                document_id=document_id,
                level=UnitLevel.FIXED,
                text=text,
                locators=_ordered_unique([token.locator for token in window]),
                order=start,
                token_count=len(window),
                metadata={"chunk_size": size, "overlap": overlap},
            )
        )
    return units


def chunk_parents(
    parents: list[RetrievalUnit], size: int, overlap: int
) -> list[RetrievalUnit]:
    children: list[RetrievalUnit] = []
    for parent in parents:
        located = [LocatedToken(token, parent.locators[0]) for token in _tokens(parent.text)]
        if not located:
            raise ValueError(f"Parent has no text and cannot produce children: {parent.unit_id}")
        parent_children = 0
        for index, (start, window) in enumerate(_windows(located, size, overlap), 1):
            children.append(
                RetrievalUnit(
                    unit_id=f"{parent.unit_id}:child:{index}",
                    document_id=parent.document_id,
                    level=UnitLevel.CHILD,
                    text=" ".join(token.text for token in window),
                    locators=parent.locators,
                    order=parent.order * 100_000 + start,
                    parent_id=parent.unit_id,
                    token_count=len(window),
                    metadata={"chunk_size": size, "overlap": overlap},
                )
            )
            parent_children += 1
        if parent_children == 0:
            raise ValueError(f"Parent without child: {parent.unit_id}")
    if len({unit.unit_id for unit in children}) != len(children):
        raise ValueError("Duplicate child unit_id detected")
    return children
