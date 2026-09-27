from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

from storage.io import atomic_write_json, sha256_file

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "data/final_dataset"
RUN_DIR = ROOT / "artifacts/final_dataset_v1/retrieval_dry_run"
OUTPUT = ROOT / "artifacts/final_dataset_v1/verification.json"


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def verify() -> dict:
    failures: list[str] = []

    validation = _load_json(DATASET_DIR / "validation_report.json")
    if validation.get("status") != "valid_author_accepted_final":
        failures.append("Dataset validation status is not final and valid")
    if validation.get("automatic_failures"):
        failures.append("Dataset validation contains automatic failures")
    if validation.get("manual_quality_flags"):
        failures.append("Dataset validation contains quality flags")

    expected_checksums = _load_json(DATASET_DIR / "checksums.json")
    checksum_mismatches: dict[str, dict[str, str | None]] = {}
    for name, expected in expected_checksums.items():
        path = DATASET_DIR / name
        actual = sha256_file(path) if path.is_file() else None
        if actual != expected:
            checksum_mismatches[name] = {
                "expected": expected,
                "actual": actual,
            }
    if checksum_mismatches:
        failures.append("Dataset checksum mismatch")

    raw_path = RUN_DIR / "raw/pilot_retrieval_runs.jsonl"
    raw_records = [
        json.loads(line)
        for line in raw_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    baseline_counts = Counter(record["baseline"] for record in raw_records)
    status_counts = Counter(record["status"] for record in raw_records)
    fingerprints = sorted({record["configuration_hash"] for record in raw_records})
    run_ids = sorted({record["run_id"] for record in raw_records})
    generation_calls = sum(
        bool(record.get("metadata", {}).get("generation_invoked"))
        for record in raw_records
    )
    retrieval_only_count = sum(
        record.get("metadata", {}).get("retrieval_only") is True
        for record in raw_records
    )
    expected_question_ids = {
        json.loads(line)["question_id"]
        for line in (DATASET_DIR / "test_questions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    }
    expected_pairs = {
        (question_id, f"B{index}")
        for question_id in expected_question_ids
        for index in range(5)
    }
    pair_counts = Counter(
        (record["question_id"], record["baseline"]) for record in raw_records
    )
    actual_pairs = set(pair_counts)
    duplicate_pairs = sorted(
        f"{question_id}:{baseline}"
        for (question_id, baseline), count in pair_counts.items()
        if count != 1
    )
    missing_pairs = sorted(
        f"{question_id}:{baseline}"
        for question_id, baseline in expected_pairs - actual_pairs
    )
    unexpected_pairs = sorted(
        f"{question_id}:{baseline}"
        for question_id, baseline in actual_pairs - expected_pairs
    )
    errors = _load_json(RUN_DIR / "metrics/pilot_errors.json")
    environment = _load_json(RUN_DIR / "environment.json")

    if len(raw_records) != 800:
        failures.append(f"Expected 800 raw records; found {len(raw_records)}")
    if baseline_counts != Counter({f"B{index}": 160 for index in range(5)}):
        failures.append(f"Unexpected baseline counts: {dict(baseline_counts)}")
    if status_counts != Counter({"completed": 800}):
        failures.append(f"Unexpected status counts: {dict(status_counts)}")
    if len(fingerprints) != 1:
        failures.append(f"Expected one configuration fingerprint: {fingerprints}")
    elif fingerprints[0] != environment.get("configuration_hash"):
        failures.append("Raw and environment configuration fingerprints differ")
    if run_ids != ["final_dataset_v1_retrieval_dry_run"]:
        failures.append(f"Unexpected run IDs: {run_ids}")
    if generation_calls:
        failures.append(f"Retrieval-only output contains {generation_calls} model calls")
    if retrieval_only_count != 800:
        failures.append(
            f"Expected 800 retrieval_only=true records; found {retrieval_only_count}"
        )
    if len(expected_question_ids) != 160:
        failures.append(
            f"Expected 160 unique test question IDs; found {len(expected_question_ids)}"
        )
    if duplicate_pairs:
        failures.append(
            f"Question-baseline pairs are not unique: {duplicate_pairs[:8]}"
        )
    if missing_pairs:
        failures.append(f"Missing question-baseline pairs: {missing_pairs[:8]}")
    if unexpected_pairs:
        failures.append(f"Unexpected question-baseline pairs: {unexpected_pairs[:8]}")
    if errors:
        failures.append(f"Experiment error file is nonempty: {len(errors)}")

    summary_rows = list(
        csv.DictReader(
            (RUN_DIR / "metrics/pilot_summary.csv").open(encoding="utf-8")
        )
    )
    if {row["baseline"] for row in summary_rows} != {
        "B0",
        "B1",
        "B2",
        "B3",
        "B4",
    }:
        failures.append("Summary does not contain exactly B0-B4")

    report_text = (RUN_DIR / "tables/pilot_metrics.md").read_text(encoding="utf-8")
    if re.search(r"\bnan\b", report_text, flags=re.IGNORECASE):
        failures.append("Rendered report still contains NaN")
    if "N/A" not in report_text:
        failures.append("Rendered retrieval-only report does not use N/A")
    if "Dataset belum dibekukan" in report_text:
        failures.append("Rendered final report still carries the draft disclaimer")

    threshold_report = _load_json(RUN_DIR / "threshold_calibration.json")
    families = threshold_report.get("families", {})
    if set(families) != {"fixed", "child_family"}:
        failures.append("Threshold families are incomplete or mixed")

    return {
        "status": "verified" if not failures else "failed",
        "dataset_status": validation.get("status"),
        "dataset_checksum_mismatches": checksum_mismatches,
        "raw_record_count": len(raw_records),
        "baseline_counts": dict(sorted(baseline_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "configuration_fingerprints": fingerprints,
        "run_ids": run_ids,
        "generation_invoked_count": generation_calls,
        "retrieval_only_true_count": retrieval_only_count,
        "unique_test_question_count": len(expected_question_ids),
        "unique_question_baseline_pair_count": len(actual_pairs),
        "duplicate_question_baseline_pairs": duplicate_pairs,
        "missing_question_baseline_pairs": missing_pairs,
        "unexpected_question_baseline_pairs": unexpected_pairs,
        "error_count": len(errors),
        "report_uses_na": "N/A" in report_text,
        "report_contains_nan": bool(
            re.search(r"\bnan\b", report_text, flags=re.IGNORECASE)
        ),
        "threshold_families": sorted(families),
        "failures": failures,
    }


def main() -> None:
    report = verify()
    atomic_write_json(OUTPUT, report)
    print(json.dumps(report, indent=2))
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
