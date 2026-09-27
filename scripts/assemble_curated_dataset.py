from __future__ import annotations

import argparse
import json
from pathlib import Path

from storage.models import QuestionRecord


def _load_seed(path: Path) -> list[QuestionRecord]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("records")
    if not isinstance(raw, list):
        raise ValueError(f"{path} must contain a JSON list or a records list")
    return [QuestionRecord.model_validate(item) for item in raw]


def _write_jsonl(path: Path, records: list[QuestionRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            record.model_dump_json(exclude_none=True) + "\n"
            for record in sorted(records, key=lambda item: item.question_id)
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble independently curated domain seeds into draft JSONL files."
    )
    parser.add_argument(
        "--ai-seed",
        type=Path,
        default=Path("data/annotations/curation/ai_curated_seed.json"),
    )
    parser.add_argument(
        "--ds-seed",
        type=Path,
        default=Path("data/annotations/curation/ds_curated_seed.json"),
    )
    parser.add_argument(
        "--development-output",
        type=Path,
        default=Path("data/annotations/final_development_questions.draft.jsonl"),
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=Path("data/annotations/final_test_questions.draft.jsonl"),
    )
    args = parser.parse_args()

    records = _load_seed(args.ai_seed) + _load_seed(args.ds_seed)
    ids = [record.question_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Curated seeds contain duplicate question IDs")
    if any(
        record.annotation_status != "draft_pending_human_verification"
        for record in records
    ):
        raise ValueError("Curated records must remain pending human verification")

    development = [record for record in records if record.split == "development"]
    test = [record for record in records if record.split == "test"]
    if len(development) != 20 or len(test) != 160:
        raise ValueError(
            f"Expected 20 development and 160 test records; "
            f"found {len(development)} and {len(test)}"
        )

    _write_jsonl(args.development_output, development)
    _write_jsonl(args.test_output, test)
    print(
        json.dumps(
            {
                "status": "draft_pending_human_verification",
                "development_questions": len(development),
                "test_questions": len(test),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
