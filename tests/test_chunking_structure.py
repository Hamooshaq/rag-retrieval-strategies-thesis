from chunking import chunk_fixed, chunk_parents
from storage.models import ParsedBlock
from structure import build_structure


def _blocks():
    return [
        ParsedBlock(
            block_id="b1",
            document_id="d1",
            locator="page:1",
            order=0,
            block_type="heading",
            text="Page one " + "alpha " * 20,
        ),
        ParsedBlock(
            block_id="b2",
            document_id="d1",
            locator="page:2",
            order=1,
            block_type="body",
            text="Page two " + "beta " * 20,
        ),
    ]


def test_children_never_cross_parent_locator():
    structure = build_structure(_blocks())
    children = chunk_parents(structure.parents, size=8, overlap=2)
    assert children
    assert all(len(child.locators) == 1 for child in children)
    assert all(child.parent_id for child in children)


def test_fixed_chunk_preserves_multiple_locators_when_crossing_boundary():
    fixed = chunk_fixed(_blocks(), size=30, overlap=0)
    assert any(len(unit.locators) == 2 for unit in fixed)


def test_neighbors_are_bidirectional_in_order():
    structure = build_structure(_blocks())
    assert structure.parents[0].neighbor_ids == [structure.parents[1].unit_id]
    assert structure.parents[1].neighbor_ids == [structure.parents[0].unit_id]
