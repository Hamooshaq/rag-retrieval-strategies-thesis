from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

QUESTION_DECISION_FIELDS = (
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
)
CORPUS_VERIFICATION_FIELDS = (
    "source_url_verified",
    "license_verified",
    "hash_verified",
    "document_opens",
    "content_scope_suitable",
    "parse_reading_order_acceptable",
    "locator_stability_acceptable",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _count_values(rows: list[dict[str, str]], column: str) -> dict[str, int]:
    return dict(sorted(Counter(row.get(column, "") for row in rows).items()))


def _human_rows_complete(rows: list[dict[str, str]]) -> bool:
    for row in rows:
        if row.get("review_status") != "SUBMITTED":
            return False
        if any(row.get(field, "") in {"", "PENDING", "UNCERTAIN"} for field in QUESTION_DECISION_FIELDS):
            return False
        if (
            row.get("question_type_agree") == "NO"
            and not row.get("reviewer_question_type")
        ):
            return False
        if (
            row.get("answerability_agree") == "NO"
            and row.get("reviewer_answerable") not in {"true", "false"}
        ):
            return False
    return True


def _resolved_adjudication_complete(row: dict[str, str]) -> bool:
    if row.get("adjudication_status") == "AGREED_NO_CHANGE":
        return True
    if row.get("adjudication_status") != "RESOLVED":
        return False
    if not (
        row.get("adjudicated_question")
        and row.get("adjudicated_question_type")
        in {"factual", "conceptual", "global", "multi_hop"}
        and row.get("adjudicated_answerable") in {"true", "false"}
        and row.get("decision_reason")
    ):
        return False
    if row["adjudicated_answerable"] == "true":
        return all(
            row.get(field)
            for field in (
                "adjudicated_reference_answer",
                "adjudicated_gold_locators",
                "adjudicated_gold_unit_ids",
                "adjudicated_gold_spans",
            )
        )
    return not any(
        row.get(field)
        for field in (
            "adjudicated_reference_answer",
            "adjudicated_gold_locators",
            "adjudicated_gold_unit_ids",
            "adjudicated_gold_spans",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only check of whether human review permits dataset freezing."
    )
    parser.add_argument(
        "--review-dir",
        type=Path,
        default=Path("data/annotations/review"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/final_dataset_draft/freeze_readiness.json"),
    )
    args = parser.parse_args()

    corpus = _read_csv(args.review_dir / "corpus_acceptance_checklist.csv")
    reviewer_a = _read_csv(args.review_dir / "student_a_pre_adjudication.csv")
    reviewer_b = _read_csv(args.review_dir / "student_b_pre_adjudication.csv")
    adjudication = _read_csv(args.review_dir / "adjudication_log.csv")
    assignments = _read_csv(args.review_dir / "cross_annotation_assignments.csv")

    blockers: list[str] = []
    if len(corpus) != 10:
        blockers.append(f"Corpus checklist contains {len(corpus)} rows, expected 10")
    if len(reviewer_a) != 180 or len(reviewer_b) != 180:
        blockers.append(
            f"Reviewer files contain A={len(reviewer_a)}, B={len(reviewer_b)} rows; "
            "expected 180 each"
        )
    if len(adjudication) != 180:
        blockers.append(
            f"Adjudication log contains {len(adjudication)} rows, expected 180"
        )
    if len(assignments) != 180:
        blockers.append(
            f"Assignment file contains {len(assignments)} rows, expected 180"
        )

    expected_question_ids = {row.get("question_id") for row in reviewer_a}
    if any(
        {row.get("question_id") for row in rows} != expected_question_ids
        for rows in (reviewer_b, adjudication, assignments)
    ):
        blockers.append("Question IDs differ across human-review files")

    corpus_ready = all(
        row.get("a_status") == "SUBMITTED"
        and row.get("b_status") == "SUBMITTED"
        and row.get("joint_decision") == "ACCEPT"
        and row.get("corpus_status") == "ACCEPTED"
        and all(
            row.get(f"a_{field}") == "YES"
            and row.get(f"b_{field}") == "YES"
            for field in CORPUS_VERIFICATION_FIELDS
        )
        for row in corpus
    )
    a_ready = _human_rows_complete(reviewer_a)
    b_ready = _human_rows_complete(reviewer_b)
    adjudication_ready = all(
        row.get("approved_by_mohammad") == "YES"
        and row.get("approved_by_fitria") == "YES"
        and row.get("adjudication_status") in {"RESOLVED", "AGREED_NO_CHANGE"}
        and row.get("freeze_eligible") == "YES"
        and _resolved_adjudication_complete(row)
        for row in adjudication
    )
    assignments_ready = all(
        row.get("reviewer_a_status") == "SUBMITTED"
        and row.get("reviewer_b_status") == "SUBMITTED"
        and row.get("adjudication_status") in {"RESOLVED", "AGREED_NO_CHANGE"}
        and row.get("freeze_eligible") == "YES"
        for row in assignments
    )

    if not corpus_ready:
        blockers.append("Ten-document corpus acceptance is incomplete")
    if not a_ready:
        blockers.append("Reviewer A's independent review is incomplete")
    if not b_ready:
        blockers.append("Reviewer B's independent review is incomplete")
    if not adjudication_ready:
        blockers.append("Joint adjudication and approval are incomplete")
    if not assignments_ready:
        blockers.append("Assignment status ledger is incomplete")

    report = {
        "status": "READY_TO_FREEZE" if not blockers else "BLOCKED_NOT_FROZEN",
        "safe_to_run_final_experiment": not blockers,
        "blockers": blockers,
        "corpus_statuses": {
            "reviewer_a": _count_values(corpus, "a_status"),
            "reviewer_b": _count_values(corpus, "b_status"),
            "joint": _count_values(corpus, "joint_decision"),
            "corpus": _count_values(corpus, "corpus_status"),
        },
        "question_review_statuses": {
            "reviewer_a": _count_values(reviewer_a, "review_status"),
            "reviewer_b": _count_values(reviewer_b, "review_status"),
            "adjudication": _count_values(adjudication, "adjudication_status"),
            "freeze_eligible": _count_values(adjudication, "freeze_eligible"),
            "assignment_freeze_eligible": _count_values(
                assignments, "freeze_eligible"
            ),
        },
        "note": (
            "This script never changes human decisions or creates final files. "
            "READY_TO_FREEZE only confirms that the required human fields were completed."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
