from __future__ import annotations

import argparse
import json
from pathlib import Path

from .runner import ExperimentRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="edu-rag",
        description="Reproducible educational PDF/PPT RAG experiment runner.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Prepare indexes and execute B0-B4")
    run_parser.add_argument("--config", required=True, type=Path)
    run_parser.add_argument("--retrieval-only", action="store_true")
    run_parser.add_argument("--retry-errors", action="store_true")

    evaluate_parser = subparsers.add_parser("evaluate", help="Rebuild metrics from raw JSONL")
    evaluate_parser.add_argument("--config", required=True, type=Path)
    evaluate_parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Rebuild reports from raw/pilot_retrieval_runs.jsonl",
    )

    validate_parser = subparsers.add_parser("validate", help="Validate manifests and annotations")
    validate_parser.add_argument("--config", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    runner = ExperimentRunner(args.config)
    if args.command == "run":
        reports = runner.run(
            retrieval_only=args.retrieval_only,
            retry_errors=args.retry_errors,
        )
        print(json.dumps({key: str(value) for key, value in reports.items()}, indent=2))
    elif args.command == "evaluate":
        reports = runner.evaluate_existing(retrieval_only=args.retrieval_only)
        print(json.dumps({key: str(value) for key, value in reports.items()}, indent=2))
    elif args.command == "validate":
        print(
            json.dumps(
                {
                    "documents": len(runner.documents),
                    "development_questions": len(runner.development_questions),
                    "evaluation_questions": len(runner.evaluation_questions),
                    "status": "valid",
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
