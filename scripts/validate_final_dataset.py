from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from storage.io import (
    atomic_write_json,
    load_document_manifest,
    load_questions,
    sha256_file,
)
from storage.models import QuestionRecord, QuestionType

EXPECTED_CELLS = {
    ("artificial_intelligence", "pdf"),
    ("artificial_intelligence", "pptx"),
    ("data_structures_algorithms", "pdf"),
    ("data_structures_algorithms", "pptx"),
}
ANSWERABLE_TYPES = {
    QuestionType.FACTUAL,
    QuestionType.CONCEPTUAL,
    QuestionType.GLOBAL,
    QuestionType.MULTI_HOP,
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _load_locator_text(path: Path) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            key = (record["document_id"], record["locator"])
            if key in result:
                raise ValueError(f"Duplicate locator at line {line_number}: {key}")
            result[key] = record["text"]
    return result


def _cell(record: QuestionRecord) -> tuple[str, str]:
    if record.domain is None or record.document_format is None:
        raise ValueError(f"{record.question_id}: missing domain/document_format")
    return record.domain, record.document_format.value


def _question_counts(records: list[QuestionRecord]) -> dict:
    cell_counts: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        domain, document_format = _cell(record)
        label = f"{domain}:{document_format}"
        key = record.question_type.value if record.answerable else "unanswerable"
        cell_counts[label][key] += 1
    return {key: dict(sorted(value.items())) for key, value in sorted(cell_counts.items())}


def validate(
    manifest_path: Path,
    locator_text_path: Path,
    development_path: Path,
    test_path: Path,
) -> dict:
    manifest_payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    is_author_accepted_final = (
        manifest_payload.get("status") == "author_accepted_final"
    )
    documents = load_document_manifest(manifest_path)
    document_by_id = {document.document_id: document for document in documents}
    locator_text = _load_locator_text(locator_text_path)
    development = load_questions(development_path)
    test = load_questions(test_path)
    all_records = development + test

    failures: list[str] = []
    manual_review_flags: list[str] = []
    if len(development) != 20:
        failures.append(f"Expected 20 development questions; found {len(development)}")
    if len(test) != 160:
        failures.append(f"Expected 160 test questions; found {len(test)}")
    if len({record.question_id for record in all_records}) != len(all_records):
        failures.append("Question IDs overlap between development and test")

    normalized_questions = [_normalize(record.question) for record in all_records]
    duplicates = [
        question for question, count in Counter(normalized_questions).items() if count > 1
    ]
    if duplicates:
        failures.append(f"Duplicate normalized question text: {duplicates[:5]}")

    test_locator_keys = {
        (record.document_id, locator)
        for record in test
        if record.answerable
        for locator in record.gold_evidence.locators
    }
    test_spans = {
        _normalize(span)
        for record in test
        if record.answerable
        for span in record.gold_evidence.spans
    }
    locator_overlap = sorted(
        {
            (record.question_id, record.document_id, locator)
            for record in development
            if record.answerable
            for locator in record.gold_evidence.locators
            if (record.document_id, locator) in test_locator_keys
        }
    )
    span_overlap = sorted(
        {
            (record.question_id, _normalize(span))
            for record in development
            if record.answerable
            for span in record.gold_evidence.spans
            if _normalize(span) in test_spans
        }
    )
    if locator_overlap:
        failures.append(
            f"Development/test gold-locator overlap: {locator_overlap[:8]}"
        )
    if span_overlap:
        failures.append(
            f"Development/test normalized-span overlap: {span_overlap[:8]}"
        )

    for record in all_records:
        document = document_by_id.get(record.document_id)
        if document is None:
            failures.append(f"{record.question_id}: unknown document {record.document_id}")
            continue
        if record.domain != document.domain:
            failures.append(f"{record.question_id}: domain differs from manifest")
        if record.document_format != document.format:
            failures.append(f"{record.question_id}: format differs from manifest")
        expected_split = (
            "development" if record in development else "test"
        )
        if record.split != expected_split:
            failures.append(
                f"{record.question_id}: split={record.split}, expected={expected_split}"
            )
        expected_status = (
            "author_accepted"
            if is_author_accepted_final
            else "draft_pending_human_verification"
        )
        if record.annotation_status != expected_status:
            failures.append(
                f"{record.question_id}: annotation_status="
                f"{record.annotation_status}, expected={expected_status}"
            )
        if record.answerable:
            if not record.expected_keywords:
                failures.append(f"{record.question_id}: expected_keywords is empty")
            missing_keywords = [
                keyword
                for keyword in record.expected_keywords
                if _normalize(keyword) not in _normalize(record.reference_answer)
            ]
            if missing_keywords:
                failures.append(
                    f"{record.question_id}: expected keywords absent from "
                    f"reference answer: {missing_keywords}"
                )
            if record.question_type in {
                QuestionType.GLOBAL,
                QuestionType.MULTI_HOP,
            } and len(record.gold_evidence.locators) < 2:
                failures.append(
                    f"{record.question_id}: {record.question_type.value} needs >=2 locators"
                )
            expected_units = {
                f"{record.document_id}:parent:{locator}"
                for locator in record.gold_evidence.locators
            }
            if set(record.gold_evidence.unit_ids) != expected_units:
                failures.append(f"{record.question_id}: unit_ids do not match locators")
            source_parts: list[str] = []
            for locator in record.gold_evidence.locators:
                key = (record.document_id, locator)
                if key not in locator_text:
                    failures.append(f"{record.question_id}: unknown locator {locator}")
                else:
                    source_parts.append(locator_text[key])
            normalized_source = _normalize(" ".join(source_parts))
            for span in record.gold_evidence.spans:
                if _normalize(span) not in normalized_source:
                    failures.append(
                        f"{record.question_id}: evidence span not found in gold locator text"
                    )
        else:
            if record.reference_answer.strip():
                failures.append(
                    f"{record.question_id}: unanswerable reference answer must be empty"
                )
            if "verified_absent_anchors=" not in record.provenance:
                manual_review_flags.append(
                    f"{record.question_id}: absent anchors not recorded"
                )

        if re.search(
            r"\b(?:describe|about|and|are)\s+"
            r"(?:the stated concept|other|what)\s*\?$",
            record.question,
            flags=re.IGNORECASE,
        ):
            manual_review_flags.append(
                f"{record.question_id}: potentially vague wording: {record.question}"
            )
        if re.search(r"\b(and|or|of|to|with)\?$", record.question, flags=re.IGNORECASE):
            manual_review_flags.append(
                f"{record.question_id}: potentially incomplete wording: {record.question}"
            )

    test_cells = Counter(_cell(record) for record in test)
    if set(test_cells) != EXPECTED_CELLS:
        failures.append(f"Unexpected test cells: {sorted(test_cells)}")
    for cell in EXPECTED_CELLS:
        cell_records = [record for record in test if _cell(record) == cell]
        if len(cell_records) != 40:
            failures.append(f"{cell}: expected 40 test records; found {len(cell_records)}")
        answerable = [record for record in cell_records if record.answerable]
        unanswerable = [record for record in cell_records if not record.answerable]
        if len(answerable) != 32 or len(unanswerable) != 8:
            failures.append(
                f"{cell}: expected 32 answerable + 8 unanswerable; "
                f"found {len(answerable)} + {len(unanswerable)}"
            )
        type_counts = Counter(record.question_type for record in answerable)
        for question_type in ANSWERABLE_TYPES:
            if type_counts[question_type] != 8:
                failures.append(
                    f"{cell}: expected 8 answerable {question_type.value}; "
                    f"found {type_counts[question_type]}"
                )

    test_answerable_counts = Counter(
        record.question_type.value for record in test if record.answerable
    )
    if any(test_answerable_counts[item.value] != 32 for item in ANSWERABLE_TYPES):
        failures.append(
            f"Expected 32 answerable questions per type; found "
            f"{dict(test_answerable_counts)}"
        )
    if sum(not record.answerable for record in test) != 32:
        failures.append("Expected exactly 32 unanswerable test questions")

    represented_documents = Counter(record.document_id for record in all_records)
    missing_documents = sorted(set(document_by_id) - set(represented_documents))
    if missing_documents:
        failures.append(f"Documents without questions: {missing_documents}")
    for document_id in sorted(document_by_id):
        represented_types = {
            record.question_type
            for record in test
            if record.document_id == document_id and record.answerable
        }
        missing_types = sorted(
            question_type.value
            for question_type in ANSWERABLE_TYPES - represented_types
        )
        if missing_types:
            failures.append(
                f"{document_id}: no answerable test coverage for {missing_types}"
            )

    return {
        "status": (
            (
                "valid_author_accepted_final"
                if is_author_accepted_final
                else "structurally_valid_pending_human_verification"
            )
            if not failures
            else "invalid"
        ),
        "manifest_documents": len(documents),
        "development_questions": len(development),
        "test_questions": len(test),
        "test_distribution": _question_counts(test),
        "questions_per_document": dict(sorted(represented_documents.items())),
        "development_test_locator_overlap": locator_overlap,
        "development_test_span_overlap": span_overlap,
        "automatic_failures": failures,
        (
            "manual_quality_flags"
            if is_author_accepted_final
            else "manual_review_flags"
        ): manual_review_flags,
        "acceptance_basis": (
            manifest_payload.get("acceptance_basis")
            if is_author_accepted_final
            else "Project-author acceptance is required before materialization."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a dataset draft or an author-accepted final package."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/final_candidate_documents.yaml"),
    )
    parser.add_argument(
        "--locator-text",
        type=Path,
        default=Path("artifacts/final_dataset_draft/locator_text.jsonl"),
    )
    parser.add_argument(
        "--development",
        type=Path,
        default=Path("data/annotations/final_development_questions.draft.jsonl"),
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("data/annotations/final_test_questions.draft.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/final_dataset_draft/validation_report.json"),
    )
    args = parser.parse_args()
    report = validate(
        args.manifest,
        args.locator_text,
        args.development,
        args.test,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if (
        report["status"] == "valid_author_accepted_final"
        and args.output.parent == args.manifest.parent
    ):
        package_files = (
            args.manifest,
            args.development,
            args.test,
            args.manifest.parent / "dataset_metadata.json",
            args.output,
            args.manifest.parent / "DATASET_CARD.md",
        )
        atomic_write_json(
            args.manifest.parent / "checksums.json",
            {path.name: sha256_file(path) for path in package_files},
        )
    print(json.dumps(report, indent=2))
    if report["automatic_failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
