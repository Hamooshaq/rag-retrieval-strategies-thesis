"""Validate completeness and pending status of the cross-annotation package."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import yaml


REVIEWER_A = ("Reviewer A", "reviewer-a")
REVIEWER_B = ("Reviewer B", "reviewer-b")
PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?62|0)8\d{8,12}(?!\d)")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _assert_pending(rows: list[dict[str, str]], columns: tuple[str, ...], label: str) -> None:
    for column in columns:
        _require(all(row[column] == "PENDING" for row in rows), f"{label}.{column} not pending")


def validate(
    package_dir: Path,
    manifest_path: Path,
    development_path: Path,
    test_path: Path,
) -> None:
    development = _read_jsonl(development_path)
    test = _read_jsonl(test_path)
    records = development + test
    expected_ids = {str(record["question_id"]) for record in records}
    _require(len(development) == 20, "Development draft must contain 20 questions")
    _require(len(test) == 160, "Test draft must contain 160 questions")
    _require(len(expected_ids) == 180, "Draft must contain 180 unique question IDs")

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    expected_document_ids = {
        str(document["document_id"]) for document in manifest["documents"]
    }
    _require(len(expected_document_ids) == 10, "Manifest must contain 10 unique documents")

    assignments = _read_csv(package_dir / "cross_annotation_assignments.csv")
    student_a = _read_csv(package_dir / "student_a_pre_adjudication.csv")
    student_b = _read_csv(package_dir / "student_b_pre_adjudication.csv")
    adjudication = _read_csv(package_dir / "adjudication_log.csv")
    corpus = _read_csv(package_dir / "corpus_acceptance_checklist.csv")

    for label, rows in (
        ("assignments", assignments),
        ("student_a", student_a),
        ("student_b", student_b),
        ("adjudication", adjudication),
    ):
        ids = [row["question_id"] for row in rows]
        _require(len(ids) == 180, f"{label} must contain exactly 180 rows")
        _require(len(set(ids)) == 180, f"{label} contains duplicate question IDs")
        _require(set(ids) == expected_ids, f"{label} question IDs do not match drafts")

    lead_a = sum(row["annotation_lead_nim"] == REVIEWER_A[1] for row in assignments)
    lead_b = sum(row["annotation_lead_nim"] == REVIEWER_B[1] for row in assignments)
    _require((lead_a, lead_b) == (90, 90), "Lead assignments must be split 90-90")
    _require(
        all(row["required_review_mode"] == "BLIND_INDEPENDENT" for row in assignments),
        "Every question must require blind independent review",
    )
    _require(
        all("AND" in row["required_reviewers"] for row in assignments),
        "Every question must require both reviewers",
    )
    _assert_pending(
        assignments,
        ("reviewer_a_status", "reviewer_b_status", "adjudication_status"),
        "assignments",
    )
    _require(
        all(row["freeze_eligible"] == "NO" for row in assignments),
        "No assignment may be freeze-eligible before human review",
    )

    _require(
        all(
            row["reviewer_name"] == REVIEWER_A[0] and row["reviewer_nim"] == REVIEWER_A[1]
            for row in student_a
        ),
        "Student A identity mismatch",
    )
    _require(
        all(
            row["reviewer_name"] == REVIEWER_B[0] and row["reviewer_nim"] == REVIEWER_B[1]
            for row in student_b
        ),
        "Student B identity mismatch",
    )
    student_pending_columns = (
        "source_opened",
        "license_and_source_checked",
        "question_semantically_clear",
        "question_naturalness_acceptable",
        "question_type_agree",
        "answerability_agree",
        "locator_exists",
        "locator_supports_answer",
        "reference_answer_accurate",
        "reference_answer_complete",
        "gold_spans_exact",
        "visual_required_agree",
        "ambiguity_or_leakage_flag",
        "review_status",
    )
    _assert_pending(student_a, student_pending_columns, "student_a")
    _assert_pending(student_b, student_pending_columns, "student_b")
    _assert_pending(
        adjudication,
        (
            "reviewer_a_status",
            "reviewer_b_status",
            "approved_by_mohammad",
            "approved_by_fitria",
            "adjudication_status",
        ),
        "adjudication",
    )
    _require(
        all(row["freeze_eligible"] == "NO" for row in adjudication),
        "No adjudication row may be freeze-eligible",
    )

    corpus_ids = [row["document_id"] for row in corpus]
    _require(len(corpus_ids) == 10, "Corpus checklist must contain 10 rows")
    _require(set(corpus_ids) == expected_document_ids, "Corpus checklist document mismatch")
    _assert_pending(
        corpus,
        (
            "a_source_url_verified",
            "a_license_verified",
            "a_hash_verified",
            "a_document_opens",
            "a_content_scope_suitable",
            "a_parse_reading_order_acceptable",
            "a_locator_stability_acceptable",
            "a_status",
            "b_source_url_verified",
            "b_license_verified",
            "b_hash_verified",
            "b_document_opens",
            "b_content_scope_suitable",
            "b_parse_reading_order_acceptable",
            "b_locator_stability_acceptable",
            "b_status",
            "joint_decision",
        ),
        "corpus",
    )
    _require(
        all(row["corpus_status"] == "PENDING_NOT_FROZEN" for row in corpus),
        "Corpus must remain pending and unfrozen",
    )

    for path in package_dir.iterdir():
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            _require(not PHONE_PATTERN.search(text), f"Phone number found in {path.name}")

    print("Cross-annotation package validation: PASS")
    print("180 question IDs, 10 documents, 90-90 lead split, all human statuses pending")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=Path("data/annotations/review"),
    )
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
    args = parser.parse_args()
    validate(args.package_dir, args.manifest, args.development, args.test)


if __name__ == "__main__":
    main()
