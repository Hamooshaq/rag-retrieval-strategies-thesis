from evaluation.metrics import evaluate_record, router_classification_metrics
from storage.models import ContextPack, QuestionRecord, RetrievalHit, RunRecord


def _question(answerable=True):
    return QuestionRecord.model_validate(
        {
            "question_id": "q1",
            "document_id": "d1",
            "question": "What is alpha?",
            "question_type": "factual",
            "answerable": answerable,
            "reference_answer": "alpha" if answerable else "",
            "expected_keywords": ["alpha"] if answerable else [],
            "gold_evidence": {"locators": ["page:2"] if answerable else []},
            "provenance": "test",
        }
    )


def _record(output="alpha [page:2]"):
    hit = RetrievalHit(
        unit_id="u",
        document_id="d1",
        level="child",
        rank=1,
        score=0.9,
        text="alpha",
        locators=["page:2"],
        parent_id="p",
    )
    return RunRecord(
        run_id="r",
        configuration_hash="a" * 64,
        question_id="q1",
        document_id="d1",
        baseline="B3",
        status="completed",
        seed=1,
        question="What is alpha?",
        question_type_gold="factual",
        question_type_predicted="factual",
        route="child",
        answerable=True,
        gold_locators=["page:2"],
        ranking=[hit],
        threshold=0.2,
        context=ContextPack(
            text="[page:2] alpha",
            token_count=2,
            locators=["page:2"],
            unit_ids=["u"],
        ),
        prompt="prompt",
        output=output,
        citations=["[page:2]"],
        retrieval_latency_ms=1,
        generation_latency_ms=2,
        end_to_end_latency_ms=3,
        prompt_tokens=4,
        output_tokens=2,
        embedding_model="hash",
        generator_model="extractive",
        document_sha256="b" * 64,
    )


def test_retrieval_and_citation_metrics():
    row = evaluate_record(_record(), _question(), top_k=4)
    assert row["recall_at_k"] == 1
    assert row["mrr"] == 1
    assert 0 <= row["ndcg_at_k"] <= 1
    assert row["source_location_accuracy"] == 1
    assert row["citation_precision"] == 1


def test_router_macro_f1():
    result = router_classification_metrics([_record()])
    assert result["macro_f1"] == 1
    assert result["n"] == 1


def test_locator_metrics_support_multi_locator_partial_coverage():
    record = _record()
    record.ranking[0].locators = ["page:2", "page:3"]
    record.context.locators = ["page:2", "page:3"]
    question = _question()
    question.gold_evidence.locators = ["page:2", "page:4"]
    row = evaluate_record(record, question, top_k=4)
    assert row["recall_at_k"] == 0.5
    assert row["precision_at_k"] == 0.5
    assert row["all_evidence_coverage"] == 0
    assert 0 <= row["ndcg_at_k"] <= 1


def test_locator_metrics_use_best_preannotated_alternative_evidence_set():
    record = _record()
    question = _question()
    question.gold_evidence.locators = ["page:8"]
    question.gold_evidence.alternative_locator_sets = [["page:2"]]
    row = evaluate_record(record, question, top_k=4)
    assert row["recall_at_k"] == 1
    assert row["source_location_accuracy"] == 1
    assert row["citation_recall"] == 1


def test_retrieval_only_records_do_not_produce_answer_metrics():
    record = _record(output="RETRIEVAL_ONLY")
    record.metadata["retrieval_only"] = True
    row = evaluate_record(record, _question(), top_k=4)
    for metric in (
        "correct_refusal",
        "false_refusal",
        "unsupported_answer",
        "reference_keyword_recall",
        "citation_precision",
        "citation_recall",
    ):
        assert row[metric] is None
