from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from storage.models import ParsedBlock, RetrievalUnit, UnitLevel


def count_whitespace_tokens(text: str) -> int:
    return len(text.split())


@dataclass(frozen=True)
class StructureBundle:
    parents: list[RetrievalUnit]
    parent_by_id: dict[str, RetrievalUnit]
    parent_order_by_document: dict[str, list[str]]


def build_structure(blocks: list[ParsedBlock]) -> StructureBundle:
    if not blocks:
        raise ValueError("Cannot build structure from an empty block list")
    grouped: dict[tuple[str, str], list[ParsedBlock]] = defaultdict(list)
    for block in blocks:
        if not block.locator:
            raise ValueError(f"Missing locator for block {block.block_id}")
        grouped[(block.document_id, block.locator)].append(block)

    parents: list[RetrievalUnit] = []
    for (document_id, locator), locator_blocks in grouped.items():
        locator_blocks.sort(key=lambda block: block.order)
        text = "\n".join(block.text for block in locator_blocks if block.text.strip()).strip()
        if not text:
            continue
        first_order = min(block.order for block in locator_blocks)
        parent_id = f"{document_id}:parent:{locator}"
        parents.append(
            RetrievalUnit(
                unit_id=parent_id,
                document_id=document_id,
                level=UnitLevel.PARENT,
                text=text,
                locators=[locator],
                order=first_order,
                token_count=count_whitespace_tokens(text),
                metadata={
                    "block_ids": [block.block_id for block in locator_blocks],
                    "block_types": [block.block_type for block in locator_blocks],
                },
            )
        )
    if not parents:
        raise ValueError("No non-empty parent units were created")

    parents.sort(key=lambda unit: (unit.document_id, unit.order))
    by_document: dict[str, list[RetrievalUnit]] = defaultdict(list)
    for parent in parents:
        by_document[parent.document_id].append(parent)

    enriched: list[RetrievalUnit] = []
    order_map: dict[str, list[str]] = {}
    for document_id, units in by_document.items():
        order_map[document_id] = [unit.unit_id for unit in units]
        for index, unit in enumerate(units):
            neighbors: list[str] = []
            if index > 0:
                neighbors.append(units[index - 1].unit_id)
            if index + 1 < len(units):
                neighbors.append(units[index + 1].unit_id)
            enriched.append(unit.model_copy(update={"neighbor_ids": neighbors}))
    enriched.sort(key=lambda unit: (unit.document_id, unit.order))
    return StructureBundle(
        parents=enriched,
        parent_by_id={unit.unit_id: unit for unit in enriched},
        parent_order_by_document=order_map,
    )
