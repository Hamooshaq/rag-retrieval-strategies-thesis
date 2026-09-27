"""Build the pending cross-annotation package for the final-dataset draft.

The generated files are workflow templates only. This script never marks a
human verification, adjudication, or corpus acceptance task as completed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml


REVIEWER_A = ("Reviewer A", "reviewer-a")
REVIEWER_B = ("Reviewer B", "reviewer-b")
PENDING = "PENDING"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return records


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _validate_inputs(
    development: list[dict[str, Any]],
    test: list[dict[str, Any]],
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(development) != 20:
        raise ValueError(f"Expected 20 development questions, found {len(development)}")
    if len(test) != 160:
        raise ValueError(f"Expected 160 test questions, found {len(test)}")
    if len(documents) != 10:
        raise ValueError(f"Expected 10 manifest documents, found {len(documents)}")

    records = development + test
    question_ids = [str(record["question_id"]) for record in records]
    duplicate_ids = sorted({item for item in question_ids if question_ids.count(item) > 1})
    if duplicate_ids:
        raise ValueError(f"Duplicate question IDs: {duplicate_ids}")

    document_ids = {str(document["document_id"]) for document in documents}
    unknown_documents = sorted(
        {
            str(record["document_id"])
            for record in records
            if str(record["document_id"]) not in document_ids
        }
    )
    if unknown_documents:
        raise ValueError(f"Questions reference unknown documents: {unknown_documents}")

    for record in records:
        if record.get("annotation_status") != "draft_pending_human_verification":
            raise ValueError(
                f"{record['question_id']} is not pending human verification: "
                f"{record.get('annotation_status')}"
            )
    return records


def _assignment_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        lead_name, lead_nim = REVIEWER_A if index % 2 == 0 else REVIEWER_B
        cross_name, cross_nim = REVIEWER_B if index % 2 == 0 else REVIEWER_A
        rows.append(
            {
                "question_id": record["question_id"],
                "split": record["split"],
                "document_id": record["document_id"],
                "domain": record["domain"],
                "document_format": record["document_format"],
                "question_type_draft": record["question_type"],
                "answerable_draft": str(bool(record["answerable"])).lower(),
                "annotation_lead_name": lead_name,
                "annotation_lead_nim": lead_nim,
                "blind_cross_reviewer_name": cross_name,
                "blind_cross_reviewer_nim": cross_nim,
                "required_reviewers": "Reviewer A AND Reviewer B",
                "required_review_mode": "BLIND_INDEPENDENT",
                "disclosure_rule": "Do not view or exchange reviewer decisions before both submit",
                "reviewer_a_status": PENDING,
                "reviewer_b_status": PENDING,
                "adjudication_status": PENDING,
                "freeze_eligible": "NO",
            }
        )
    return rows


def _student_rows(
    records: list[dict[str, Any]],
    documents_by_id: dict[str, dict[str, Any]],
    reviewer: tuple[str, str],
) -> list[dict[str, Any]]:
    reviewer_name, reviewer_nim = reviewer
    rows: list[dict[str, Any]] = []
    for record in records:
        document = documents_by_id[str(record["document_id"])]
        evidence = record["gold_evidence"]
        rows.append(
            {
                "question_id": record["question_id"],
                "split": record["split"],
                "document_id": record["document_id"],
                "domain": record["domain"],
                "document_format": record["document_format"],
                "source_path": document["path"],
                "source_url": document["source_url"],
                "question_draft": record["question"],
                "question_type_draft": record["question_type"],
                "difficulty_draft": record["difficulty_tag"],
                "answerable_draft": str(bool(record["answerable"])).lower(),
                "reference_answer_draft": record.get("reference_answer", ""),
                "gold_locators_draft": _json_cell(evidence.get("locators", [])),
                "gold_unit_ids_draft": _json_cell(evidence.get("unit_ids", [])),
                "gold_spans_draft": _json_cell(evidence.get("spans", [])),
                "alternative_locator_sets_draft": _json_cell(
                    evidence.get("alternative_locator_sets", [])
                ),
                "visual_required_draft": str(bool(record.get("visual_required", False))).lower(),
                "source_opened": PENDING,
                "license_and_source_checked": PENDING,
                "question_semantically_clear": PENDING,
                "question_naturalness_acceptable": PENDING,
                "question_type_agree": PENDING,
                "reviewer_question_type": "",
                "answerability_agree": PENDING,
                "reviewer_answerable": "",
                "locator_exists": PENDING,
                "locator_supports_answer": PENDING,
                "alternative_locators": "",
                "reference_answer_accurate": PENDING,
                "reference_answer_complete": PENDING,
                "gold_spans_exact": PENDING,
                "visual_required_agree": PENDING,
                "ambiguity_or_leakage_flag": PENDING,
                "reviewer_revised_question": "",
                "reviewer_revised_reference_answer": "",
                "reviewer_notes": "",
                "review_status": PENDING,
                "reviewer_name": reviewer_name,
                "reviewer_nim": reviewer_nim,
                "submitted_at": "",
            }
        )
    return rows


def _adjudication_rows(
    records: list[dict[str, Any]], assignment_by_id: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        assignment = assignment_by_id[str(record["question_id"])]
        rows.append(
            {
                "question_id": record["question_id"],
                "split": record["split"],
                "document_id": record["document_id"],
                "domain": record["domain"],
                "document_format": record["document_format"],
                "question_type_draft": record["question_type"],
                "answerable_draft": str(bool(record["answerable"])).lower(),
                "disagreement_summary_lead": assignment["annotation_lead_name"],
                "reviewer_a_status": PENDING,
                "reviewer_b_status": PENDING,
                "disagreement_fields": "",
                "adjudicated_question": "",
                "adjudicated_question_type": "",
                "adjudicated_answerable": "",
                "adjudicated_reference_answer": "",
                "adjudicated_gold_locators": "",
                "adjudicated_gold_unit_ids": "",
                "adjudicated_gold_spans": "",
                "adjudicated_visual_required": "",
                "decision_reason": "",
                "approved_by_mohammad": PENDING,
                "approved_by_fitria": PENDING,
                "adjudication_status": PENDING,
                "freeze_eligible": "NO",
                "adjudicated_at": "",
            }
        )
    return rows


def _corpus_rows(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for document in documents:
        row: dict[str, Any] = {
            "document_id": document["document_id"],
            "title": document["title"],
            "domain": document["domain"],
            "document_format": document["format"],
            "source_path": document["path"],
            "source_url": document["source_url"],
            "license_declared": document["license"],
            "expected_sha256": document["sha256"],
        }
        for prefix in ("a", "b"):
            row.update(
                {
                    f"{prefix}_source_url_verified": PENDING,
                    f"{prefix}_license_verified": PENDING,
                    f"{prefix}_hash_verified": PENDING,
                    f"{prefix}_document_opens": PENDING,
                    f"{prefix}_content_scope_suitable": PENDING,
                    f"{prefix}_parse_reading_order_acceptable": PENDING,
                    f"{prefix}_locator_stability_acceptable": PENDING,
                    f"{prefix}_notes": "",
                    f"{prefix}_status": PENDING,
                }
            )
        row.update(
            {
                "joint_decision": PENDING,
                "joint_notes": "",
                "corpus_status": "PENDING_NOT_FROZEN",
            }
        )
        rows.append(row)
    return rows


def build_package(
    manifest_path: Path,
    development_path: Path,
    test_path: Path,
    output_dir: Path,
) -> None:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    documents = list(manifest["documents"])
    development = _read_jsonl(development_path)
    test = _read_jsonl(test_path)
    records = _validate_inputs(development, test, documents)

    documents_by_id = {str(item["document_id"]): item for item in documents}
    assignments = _assignment_rows(records)
    assignments_by_id = {str(item["question_id"]): item for item in assignments}

    _write_csv(
        output_dir / "cross_annotation_assignments.csv",
        list(assignments[0]),
        assignments,
    )
    student_a = _student_rows(records, documents_by_id, REVIEWER_A)
    student_b = _student_rows(records, documents_by_id, REVIEWER_B)
    _write_csv(output_dir / "student_a_pre_adjudication.csv", list(student_a[0]), student_a)
    _write_csv(output_dir / "student_b_pre_adjudication.csv", list(student_b[0]), student_b)

    adjudication = _adjudication_rows(records, assignments_by_id)
    _write_csv(output_dir / "adjudication_log.csv", list(adjudication[0]), adjudication)

    corpus = _corpus_rows(documents)
    _write_csv(output_dir / "corpus_acceptance_checklist.csv", list(corpus[0]), corpus)

    print(f"Created pending cross-annotation package in {output_dir}")
    print(f"Questions: {len(records)} (development={len(development)}, test={len(test)})")
    print("Both reviewers required for every question; lead assignments: 90 each")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/final_candidate_documents.yaml"),
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
        default=Path("data/annotations/review"),
    )
    args = parser.parse_args()
    build_package(args.manifest, args.development, args.test, args.output_dir)


if __name__ == "__main__":
    main()
