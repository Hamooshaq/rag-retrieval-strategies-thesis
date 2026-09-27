from generation import ContextBuilder, WhitespaceTokenCounter
from retrieval import ExactCosineIndex, HashingEmbedder, RoutedRetriever
from routing import OracleRouter, RuleRouter
from storage.models import QuestionRecord, RetrievalUnit


def _units():
    parents = [
        RetrievalUnit(
            unit_id="d:parent:page:1",
            document_id="d",
            level="parent",
            text="alpha evidence about locators and audit",
            locators=["page:1"],
            order=0,
            token_count=6,
            neighbor_ids=["d:parent:page:2"],
        ),
        RetrievalUnit(
            unit_id="d:parent:page:2",
            document_id="d",
            level="parent",
            text="beta evidence about thresholds and refusal",
            locators=["page:2"],
            order=1,
            token_count=6,
            neighbor_ids=["d:parent:page:1"],
        ),
    ]
    children = [
        RetrievalUnit(
            unit_id="d:parent:page:1:child:1",
            document_id="d",
            level="child",
            text="alpha locator audit",
            locators=["page:1"],
            order=0,
            parent_id=parents[0].unit_id,
            token_count=3,
        ),
        RetrievalUnit(
            unit_id="d:parent:page:2:child:1",
            document_id="d",
            level="child",
            text="beta threshold refusal",
            locators=["page:2"],
            order=1,
            parent_id=parents[1].unit_id,
            token_count=3,
        ),
    ]
    return parents, children


def test_exact_cosine_and_parent_mapping():
    parents, children = _units()
    index = ExactCosineIndex(children, HashingEmbedder())
    routed = RoutedRetriever(index, {parent.unit_id: parent for parent in parents})
    hits = routed.retrieve("alpha audit", "d", "parent", top_k=1, candidate_k=2)
    assert hits[0].unit_id == parents[0].unit_id
    assert hits[0].locators == ["page:1"]


def test_context_builder_never_exceeds_budget():
    _, children = _units()
    index = ExactCosineIndex(children, HashingEmbedder())
    hits = index.search("alpha beta", top_k=2)
    pack = ContextBuilder(WhitespaceTokenCounter(), max_tokens=5).build(
        hits, threshold=-1
    )
    assert pack.token_count <= 5
    assert pack.truncated


class SeparatorAwareCounter:
    """Joining evidence pieces contributes tokens beyond per-piece counts."""

    def count(self, text: str) -> int:
        return len(text.split()) + text.count("\n\n") * 2

    def truncate(self, text: str, max_tokens: int) -> str:
        return " ".join(text.replace("\n\n", " ").split()[:max_tokens])


def test_context_builder_checks_combined_text_budget():
    _, children = _units()
    index = ExactCosineIndex(children, HashingEmbedder())
    hits = index.search("alpha beta", top_k=2)
    pack = ContextBuilder(SeparatorAwareCounter(), max_tokens=7).build(
        hits, threshold=-1
    )
    assert pack.token_count <= 7
    assert pack.truncated


def test_context_builder_does_not_keep_locator_only_truncated_hit():
    _, children = _units()
    hits = [
        ExactCosineIndex(children, HashingEmbedder()).search(
            "alpha", top_k=1
        )[0],
        ExactCosineIndex(children, HashingEmbedder()).search(
            "beta", top_k=1
        )[0],
    ]
    pack = ContextBuilder(WhitespaceTokenCounter(), max_tokens=4).build(
        hits, threshold=-1
    )
    assert pack.unit_ids == [hits[0].unit_id]
    assert pack.locators == hits[0].locators
    assert f"[{hits[1].locators[0]}]" not in pack.text


def test_context_builder_keeps_all_locators_when_partial_body_is_present():
    multi_locator = RetrievalUnit(
        unit_id="d:fixed:1",
        document_id="d",
        level="fixed",
        text="alpha beta gamma delta",
        locators=["page:1", "page:2"],
        order=0,
        token_count=4,
    )
    hit = ExactCosineIndex([multi_locator], HashingEmbedder()).search(
        "alpha", top_k=1
    )[0]
    pack = ContextBuilder(WhitespaceTokenCounter(), max_tokens=4).build(
        [hit], threshold=-1
    )
    assert pack.unit_ids == [hit.unit_id]
    assert pack.locators == ["page:1", "page:2"]
    assert "alpha" in pack.text


def test_rule_router_and_oracle_are_distinct():
    predicted = RuleRouter().predict("Summarize the main themes across the document")
    assert predicted.route == "parent_neighbor"
    question = QuestionRecord.model_validate(
        {
            "question_id": "q",
            "document_id": "d",
            "question": "What is combined?",
            "question_type": "multi_hop",
            "answerable": True,
            "reference_answer": "x",
            "gold_evidence": {"locators": ["page:1", "page:2"]},
            "provenance": "test",
        }
    )
    oracle = OracleRouter().predict(question)
    assert oracle.route == "distinct_parents"


def test_rule_router_recognizes_two_concept_how_question_as_multi_hop():
    decision = RuleRouter().predict(
        "How do heap insertion and deletion support heap sort's runtime?"
    )
    assert decision.question_type == "multi_hop"
    assert decision.route == "distinct_parents"
