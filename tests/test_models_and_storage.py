import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from storage.io import JsonlResultStore
from storage.models import (
    ContextPack,
    DifficultyTag,
    DocumentFormat,
    GoldEvidence,
    HumanRating,
    QuestionRecord,
    QuestionType,
    RunRecord,
)


def test_answerable_requires_gold_locator():
    with pytest.raises(ValidationError):
        QuestionRecord(
            question_id="q1",
            document_id="d1",
            question="What is the value?",
            question_type=QuestionType.FACTUAL,
            answerable=True,
            reference_answer="A value.",
            gold_evidence=GoldEvidence(),
            provenance="test",
        )


def _record(
    status: str = "completed",
    configuration_hash: str = "a" * 64,
) -> RunRecord:
    return RunRecord(
        run_id="r1",
        configuration_hash=configuration_hash,
        question_id="q1",
        document_id="d1",
        baseline="B0",
        status=status,
        seed=1,
        question="Question?",
        question_type_gold="factual",
        answerable=False,
        context=ContextPack(text="", token_count=0),
        prompt="p",
        output="INSUFFICIENT_EVIDENCE",
        retrieval_latency_ms=0,
        generation_latency_ms=0,
        end_to_end_latency_ms=0,
        prompt_tokens=1,
        output_tokens=1,
        embedding_model="hash",
        document_sha256="b" * 64,
    )


def test_result_store_rejects_completed_duplicate(tmp_path: Path):
    store = JsonlResultStore(tmp_path / "runs.jsonl")
    store.append(_record())
    with pytest.raises(ValueError):
        store.append(_record())


def test_result_store_can_replace_error(tmp_path: Path):
    store = JsonlResultStore(tmp_path / "runs.jsonl")
    store.append(_record(status="error"))
    store.append(_record(status="completed"))
    reloaded = JsonlResultStore(tmp_path / "runs.jsonl")
    assert len(reloaded.records) == 1
    assert reloaded.records[0].status == "completed"


def test_result_store_keeps_same_question_from_distinct_input_fingerprints(
    tmp_path: Path,
):
    store = JsonlResultStore(tmp_path / "runs.jsonl")
    store.append(_record(configuration_hash="a" * 64))
    store.append(_record(configuration_hash="c" * 64))
    assert len(store.records) == 2


def test_annotation_schema_and_pilot_records_remain_aligned():
    project_root = Path(__file__).parents[1]
    schema = json.loads(
        (project_root / "data" / "annotations" / "schema.json").read_text(encoding="utf-8")
    )
    assert set(schema["properties"]["question_type"]["enum"]) == {
        member.value for member in QuestionType
    }
    assert {
        value for value in schema["properties"]["difficulty_tag"]["enum"] if value
    } == {member.value for member in DifficultyTag}

    for filename in ("pilot_dev_questions.jsonl", "pilot_questions.jsonl"):
        path = project_root / "data" / "annotations" / filename
        records = [
            QuestionRecord.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert records


def test_final_annotation_metadata_and_unit_ids_are_supported():
    record = QuestionRecord(
        question_id="final-q1",
        document_id="document-1",
        domain="artificial_intelligence",
        document_format=DocumentFormat.PDF,
        question="What concept is defined?",
        question_type=QuestionType.CONCEPTUAL,
        difficulty_tag=DifficultyTag.MEDIUM,
        answerable=True,
        reference_answer="The concept is defined by the cited evidence.",
        gold_evidence=GoldEvidence(
            locators=["page:2"],
            unit_ids=["document-1:parent:page:2"],
        ),
        provenance="machine-assisted draft pending two-student verification",
        annotation_status="draft_pending_human_verification",
    )
    assert record.gold_evidence.unit_ids == ["document-1:parent:page:2"]


def test_human_rating_schema_and_model_enforce_ordinal_scale():
    project_root = Path(__file__).parents[1]
    schema = json.loads(
        (
            project_root
            / "data"
            / "annotations"
            / "human_evaluation_schema.json"
        ).read_text(encoding="utf-8")
    )
    assert schema["$defs"]["OrdinalScore"] == {
        "type": "integer",
        "minimum": 0,
        "maximum": 2,
    }
    rating = HumanRating(
        rating_id="q1-a-rater-a",
        question_id="q1",
        anonymous_answer_id="answer-a",
        rater_id="A",
        answerable=True,
        refused=False,
        correctness=2,
        faithfulness=2,
        completeness=1,
        relevance=2,
        citation_correctness=2,
        evidence_locators_seen=["page:2"],
        rationale="Jawaban didukung evidence utama.",
    )
    assert rating.phase == "pre_adjudication"
    with pytest.raises(ValidationError):
        HumanRating.model_validate({**rating.model_dump(), "correctness": 3})
