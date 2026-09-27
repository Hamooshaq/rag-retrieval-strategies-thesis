from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import yaml

from storage.io import atomic_write_json, load_document_manifest, load_questions, sha256_file
from storage.models import QuestionRecord
from validate_final_dataset import validate

ACCEPTANCE_NOTE = (
    "Accepted for research use based on the project authors' attestation and "
    "automated structural validation."
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _assert_development_test_disjoint(
    development: list[QuestionRecord],
    test: list[QuestionRecord],
) -> None:
    test_locators = {
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
            if (record.document_id, locator) in test_locators
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
    if locator_overlap or span_overlap:
        raise ValueError(
            "Development/test leakage remains: "
            f"locator_overlap={locator_overlap[:8]}, span_overlap={span_overlap[:8]}"
        )


def _scope_label(record: QuestionRecord) -> str | None:
    if record.answerable:
        return None
    if record.split == "development":
        return "same_domain_near_miss"
    return (
        "same_domain_near_miss"
        if re.search(r"-u-(?:f|c|g|m)02$", record.question_id)
        else "out_of_domain"
    )


def _accepted_payload(record: QuestionRecord) -> dict:
    payload = record.model_dump(mode="json", exclude_none=True)
    payload["annotation_status"] = "author_accepted"
    provenance = re.sub(
        r";?\s*pending independent human verification"
        r"(?: and adjudication)? by both student researchers",
        "",
        record.provenance,
        flags=re.IGNORECASE,
    )
    provenance = provenance.strip(" ;.")
    payload["provenance"] = f"{provenance}. {ACCEPTANCE_NOTE}"
    scope = _scope_label(record)
    if scope:
        payload["unanswerable_scope"] = scope
    return payload


def _write_jsonl(path: Path, records: list[QuestionRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(_accepted_payload(record), ensure_ascii=False) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a clean final package after author acceptance."
        )
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
        "--output-dir",
        type=Path,
        default=Path("data/final_dataset"),
    )
    args = parser.parse_args()

    structural = validate(
        args.manifest,
        args.locator_text,
        args.development,
        args.test,
    )
    if structural["automatic_failures"]:
        raise ValueError(
            f"Draft failed structural validation: {structural['automatic_failures']}"
        )
    development = load_questions(args.development)
    test = load_questions(args.test)
    _assert_development_test_disjoint(development, test)
    documents = load_document_manifest(args.manifest)

    output_dir = args.output_dir
    development_output = output_dir / "development_questions.jsonl"
    test_output = output_dir / "test_questions.jsonl"
    _write_jsonl(development_output, development)
    _write_jsonl(test_output, test)

    raw_manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    final_manifest = {
        "schema_version": raw_manifest.get("schema_version", "1.0"),
        "dataset_version": "1.0.0",
        "status": "author_accepted_final",
        "accepted_at": datetime.now(UTC).isoformat(),
        "acceptance_basis": (
            "Project-author attestation plus automated structural, locator, "
            "span, distribution, and leakage validation."
        ),
        "documents": [
            document.model_dump(mode="json", exclude_none=True)
            for document in documents
        ],
    }
    manifest_output = output_dir / "documents.yaml"
    manifest_output.write_text(
        yaml.safe_dump(final_manifest, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    scope_counts = Counter(
        _scope_label(record) for record in test if not record.answerable
    )
    metadata = {
        "dataset_version": "1.0.0",
        "status": "author_accepted_final",
        "documents": len(documents),
        "development_questions": len(development),
        "test_questions": len(test),
        "answerable_test_questions": sum(record.answerable for record in test),
        "unanswerable_test_questions": sum(not record.answerable for record in test),
        "unanswerable_scope_distribution": dict(sorted(scope_counts.items())),
        "question_type_distribution_answerable_test": dict(
            sorted(
                Counter(
                    record.question_type.value
                    for record in test
                    if record.answerable
                ).items()
            )
        ),
        "acceptance_note": ACCEPTANCE_NOTE,
        "source_draft_hashes": {
            "manifest": sha256_file(args.manifest),
            "development": sha256_file(args.development),
            "test": sha256_file(args.test),
            "locator_text": sha256_file(args.locator_text),
        },
    }
    metadata_output = output_dir / "dataset_metadata.json"
    atomic_write_json(metadata_output, metadata)

    checksums = {
        path.name: sha256_file(path)
        for path in (
            manifest_output,
            development_output,
            test_output,
            metadata_output,
        )
    }
    atomic_write_json(output_dir / "checksums.json", checksums)
    print(json.dumps({**metadata, "checksums": checksums}, indent=2))


if __name__ == "__main__":
    main()
